"""#927 — the disk tier reads on a background thread, not on the compute thread."""

import threading
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

from soup_cli.utils.async_disk_source import AsyncDiskSource  # noqa: E402
from soup_cli.utils.layer_stream_runtime import DiskSource, RamSource  # noqa: E402

N_LAYERS = 4


def _shards(tmp_path: Path, n_layers: int = N_LAYERS) -> str:
    """A shard per layer, NF4-shaped: mixed uint8 and float32, plus a bf16 norm."""
    from soup_cli.utils.layer_shard import layer_shard_path

    out = tmp_path / "shards"
    out.mkdir()
    torch.manual_seed(927)
    for idx in range(n_layers):
        save_file(
            {
                "self_attn.q_proj.weight": torch.randint(
                    0, 255, (64, 32), dtype=torch.uint8
                ),
                "self_attn.q_proj.weight::absmax": torch.rand(16, dtype=torch.float32),
                "input_layernorm.weight": torch.rand(64, dtype=torch.bfloat16),
            },
            layer_shard_path(str(out), idx),
        )
    return str(out)


def _spec(shard_dir: str):
    return RamSource.layer_specs_from_shards(shard_dir, N_LAYERS)


class TestByteIdentityAgainstTheShippedSource:
    """THE gate: same bytes as DiskSource, or the change is wrong."""

    @pytest.mark.parametrize("read_ahead", [1, 2, 4])
    def test_every_tensor_matches_disk_source(self, tmp_path, read_ahead):
        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        shipped = DiskSource(shard_dir, N_LAYERS, spec)
        ours = AsyncDiskSource(
            shard_dir, N_LAYERS, spec, read_ahead=read_ahead, pin=False
        )
        try:
            for idx in range(N_LAYERS):
                for name in spec[idx]:
                    theirs = shipped.get(idx, name)
                    mine = ours.get(idx, name)
                    assert mine.dtype == theirs.dtype, (idx, name)
                    assert mine.shape == theirs.shape, (idx, name)
                    assert torch.equal(
                        mine.view(torch.uint8), theirs.view(torch.uint8)
                    ), (idx, name)
        finally:
            ours.close()
            shipped.close()

    def test_depth_is_performance_never_semantics(self, tmp_path):
        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        shallow = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=1, pin=False)
        deep = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=4, pin=False)
        try:
            for idx in range(N_LAYERS):
                for name in spec[idx]:
                    assert torch.equal(
                        shallow.get(idx, name).view(torch.uint8),
                        deep.get(idx, name).view(torch.uint8),
                    )
        finally:
            shallow.close()
            deep.close()


class TestItHoldsNoMapping:
    def test_no_safe_open_handle_is_ever_created(self, tmp_path, monkeypatch):
        """The commit charge #926 is about comes from mappings; we must hold none."""
        import safetensors

        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        opens = []

        real = safetensors.safe_open

        def counting(*args, **kwargs):
            opens.append(args[0] if args else kwargs.get("filename"))
            return real(*args, **kwargs)

        monkeypatch.setattr(safetensors, "safe_open", counting)
        source = AsyncDiskSource(shard_dir, N_LAYERS, spec, pin=False)
        try:
            for idx in range(N_LAYERS):
                source.get(idx, "input_layernorm.weight")
        finally:
            source.close()
        assert opens == [], f"mapped {len(opens)} shard(s) after all"

    def test_nbytes_reports_staging_not_the_store(self, tmp_path):
        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        source = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=2, pin=False)
        try:
            one_layer = sum(
                torch.empty(shape, dtype=getattr(torch, dtype)).numel()
                * torch.empty((), dtype=getattr(torch, dtype)).element_size()
                for shape, dtype in spec[0].values()
            )
            assert source.nbytes == 2 * one_layer
            assert source.disk_bytes > source.nbytes
        finally:
            source.close()


class TestFailuresAreLoudAndNeverHang:
    def test_a_read_error_surfaces_at_the_get_that_wanted_it(self, tmp_path):
        from soup_cli.utils.layer_shard import layer_shard_path

        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        source = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=1, pin=False)
        try:
            source.get(0, "input_layernorm.weight")
            Path(layer_shard_path(shard_dir, 3)).write_bytes(b"corrupt")
            with pytest.raises((OSError, ValueError)):
                for idx in range(1, N_LAYERS):
                    for name in spec[idx]:
                        source.get(idx, name)
        finally:
            source.close()

    def test_a_dead_reader_refuses_instead_of_blocking(self, tmp_path):
        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        source = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=1, pin=False)
        try:
            source._fail(RuntimeError("reader died"))
            done = threading.Event()
            captured = {}

            def call():
                try:
                    source.get(1, "input_layernorm.weight")
                except BaseException as exc:  # noqa: BLE001 — recorded for the assert
                    captured["exc"] = exc
                done.set()

            threading.Thread(target=call, daemon=True).start()
            assert done.wait(timeout=10), "get() blocked after the reader died"
            assert isinstance(captured.get("exc"), RuntimeError)
        finally:
            source.close()

    def test_close_is_idempotent_and_stops_the_thread(self, tmp_path):
        shard_dir = _shards(tmp_path)
        source = AsyncDiskSource(shard_dir, N_LAYERS, _spec(shard_dir), pin=False)
        source.get(0, "input_layernorm.weight")
        source.close()
        source.close()
        assert not source._thread.is_alive()
        with pytest.raises(RuntimeError, match="closed"):
            source.get(1, "input_layernorm.weight")

    def test_a_spec_that_disagrees_with_the_header_is_refused(self, tmp_path):
        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        spec[0]["input_layernorm.weight"] = ((128,), "bfloat16")
        with pytest.raises(ValueError, match="disagrees with"):
            AsyncDiskSource(shard_dir, N_LAYERS, spec, pin=False)

    def test_read_ahead_is_bounded(self, tmp_path):
        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        for bad in (0, 9):
            with pytest.raises(ValueError, match="stream_read_ahead"):
                AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=bad, pin=False)
        with pytest.raises(ValueError, match="must be an int"):
            AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=True, pin=False)


# ==========================================================================
# the hazards pin=False cannot express
# ==========================================================================
_NO_CUDA = not torch.cuda.is_available()


def _uniform_shards(tmp_path: Path, n_layers: int, size: int) -> str:
    """One tensor per layer, every byte equal to the layer index.

    A leak therefore names the layer it came from, which a random fixture
    cannot do: `torch.equal` says "different", this says "layer 6 is sitting in
    layer 0's buffer".
    """
    from soup_cli.utils.layer_shard import layer_shard_path

    out = tmp_path / "uniform"
    out.mkdir()
    for idx in range(n_layers):
        save_file(
            {"w": torch.full((size,), idx, dtype=torch.uint8)},
            layer_shard_path(str(out), idx),
        )
    return str(out)


@pytest.mark.skipif(_NO_CUDA, reason="the hazard is a CUDA copy draining out of pinned host memory")
class TestTheDeviceGetsTheLayerItAskedFor:
    """THE gate for the recycle-under-an-in-flight-copy defect.

    Every other test in this file runs ``pin=False``, where a host-to-device
    copy is synchronous and the hazard cannot exist — so the suite was blind to
    it by construction while ``pin=True`` is the default AND the production
    setting. This drives the REAL consumer (``LayerBufferPool`` +
    ``StreamPrefetcher``) and checks the bytes that reach the DEVICE.
    """

    @pytest.mark.parametrize("read_ahead", [1, 2, 4, 8])
    def test_no_layer_reaches_the_device_holding_another_layers_weights(
        self, tmp_path, read_ahead
    ):
        from soup_cli.utils.layer_shard import layer_shard_path
        from soup_cli.utils.layer_stream_runtime import LayerBufferPool, StreamPrefetcher

        n_layers = 8
        shard_dir = _uniform_shards(tmp_path, n_layers, 8 * 1024 * 1024)
        spec = RamSource.layer_specs_from_shards(shard_dir, n_layers)
        source = AsyncDiskSource(
            shard_dir, n_layers, spec, read_ahead=read_ahead, pin=True
        )
        try:
            assert source.pinned, "the hazard needs genuinely pinned staging"
            pool = LayerBufferPool(spec[0], n_buffers=2, device="cuda")
            stream = torch.cuda.Stream()
            prefetcher = StreamPrefetcher(pool, source, n_layers, stream)

            # Warm the page cache: a reader blocked on cold I/O cannot run
            # ahead far enough to overwrite anything, which would hide the bug.
            for idx in range(n_layers):
                Path(layer_shard_path(shard_dir, idx)).read_bytes()

            # Put the GPU behind. Prefetching only pays off when it is, and the
            # copy only stays in flight long enough to be clobbered when it is.
            hog = torch.randn(4096, 4096, device="cuda")
            for _ in range(200):
                hog = hog @ hog.clamp(-1, 1)

            seen = []
            prefetcher.prime()
            for idx in range(n_layers):
                buffers = pool.wait(idx)  # a GPU-side wait_event, not a host one
                prefetcher.advance(idx)  # -> load_async(idx+1) -> source.get(...)
                seen.append(buffers["w"].clone())
            torch.cuda.synchronize()

            wrong = {
                idx: torch.unique(got).tolist()
                for idx, got in enumerate(seen)
                if torch.unique(got).tolist() != [idx]
            }
            assert not wrong, (
                f"read_ahead={read_ahead}: {len(wrong)} of {n_layers} layers reached "
                f"the device holding another layer's weights — {wrong} (each value is "
                f"the layer the bytes actually came from). The staging buffer was "
                f"recycled while its copy was still draining."
            )
        finally:
            source.close()

    def test_the_same_harness_is_clean_through_the_shipped_sources(self, tmp_path):
        """A control: if this ever fails, the harness is wrong, not the source."""
        from soup_cli.utils.layer_stream_runtime import LayerBufferPool, StreamPrefetcher

        n_layers = 8
        shard_dir = _uniform_shards(tmp_path, n_layers, 8 * 1024 * 1024)
        spec = RamSource.layer_specs_from_shards(shard_dir, n_layers)
        for source in (
            DiskSource(shard_dir, n_layers, spec),
            RamSource(shard_dir, n_layers, spec, pin=True),
        ):
            pool = LayerBufferPool(spec[0], n_buffers=2, device="cuda")
            stream = torch.cuda.Stream()
            prefetcher = StreamPrefetcher(pool, source, n_layers, stream)
            hog = torch.randn(4096, 4096, device="cuda")
            for _ in range(200):
                hog = hog @ hog.clamp(-1, 1)
            seen = []
            prefetcher.prime()
            for idx in range(n_layers):
                buffers = pool.wait(idx)
                prefetcher.advance(idx)
                seen.append(buffers["w"].clone())
            torch.cuda.synchronize()
            for idx, got in enumerate(seen):
                assert torch.unique(got).tolist() == [idx], (
                    f"{type(source).__name__} layer {idx} is wrong — the harness, "
                    f"not the source under test, is at fault"
                )

    def test_pinned_is_measured_not_asserted(self, tmp_path):
        """``pinned=True`` must mean the staging really is page-locked.

        ``RamSource`` checks ``dst.is_pinned()`` because a box that hands back
        pageable memory would otherwise report the fast path while silently
        paying the ~97% -> ~79% GPU-utilisation cost of a synchronous copy.
        """
        shard_dir = _shards(tmp_path)
        source = AsyncDiskSource(shard_dir, N_LAYERS, _spec(shard_dir), pin=True)
        try:
            assert source.pinned is True
            staged = [dst for slot in source._slots for dst in slot.values()]
            assert staged, "no staging was allocated"
            assert all(dst.is_pinned() for dst in staged), (
                "pinned=True but torch returned pageable memory"
            )
        finally:
            source.close()


class TestTheReadHappensAhead:
    """A synchronous implementation passes every other test in this file.

    ``get`` returning the right bytes is necessary and not sufficient: the
    point of the whole source is that the read is OFF the compute thread and
    already done before the consumer asks. Nothing pinned that.
    """

    def test_reads_run_on_the_reader_thread_not_the_caller(self, tmp_path, monkeypatch):
        import soup_cli.utils.async_disk_source as module

        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        threads = []
        real = module.read_into

        def recording(handle, entry, tensor):
            threads.append(threading.current_thread())
            return real(handle, entry, tensor)

        monkeypatch.setattr(module, "read_into", recording)
        source = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=2, pin=False)
        try:
            source.get(0, "input_layernorm.weight")
            assert threads, "nothing was read at all"
            caller = threading.current_thread()
            offenders = sorted({t.name for t in threads if t is caller})
            assert not offenders, (
                f"the read ran on the calling thread ({offenders}) — this source "
                f"exists to keep it off the compute thread"
            )
            assert {t.name for t in threads} == {"soup-layer-reader"}
        finally:
            source.close()

    def test_the_next_layer_arrives_before_anyone_asks_for_it(self, tmp_path):
        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        source = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=2, pin=False)
        try:
            source.get(0, "input_layernorm.weight")
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and 1 not in source._slot_of:
                time.sleep(0.005)
            assert 1 in source._slot_of, (
                "layer 1 was never staged, though nothing asked for it yet — the "
                "source read on demand instead of ahead"
            )
        finally:
            source.close()


class TestTheShapeBuildSourceActuallyPasses:
    """``_build_source`` hands this source per-layer specs whose trailing
    entries are the vocabulary-sized embed / lm_head shards, with ENTIRELY
    different tensor keys (layer_stream_runtime.py, ``source_specs``). Sizing
    every staging slot from layer 0 made the reader raise ``KeyError`` on the
    reader thread at the first forward turnaround of any real model, poisoning
    the source permanently, where ``DiskSource`` simply returns the tensor.
    """

    @staticmethod
    def _hetero(tmp_path: Path):
        from soup_cli.utils.layer_shard import layer_shard_path

        out = tmp_path / "hetero"
        out.mkdir()
        torch.manual_seed(927)
        for idx in range(2):
            save_file(
                {"self_attn.q_proj.weight": torch.rand(8, 4, dtype=torch.float32)},
                layer_shard_path(str(out), idx),
            )
        big = out / "embed.safetensors"
        save_file(
            {"model.embed_tokens.weight": torch.rand(64, 4, dtype=torch.float32)},
            str(big),
        )
        specs = [
            {"self_attn.q_proj.weight": ((8, 4), "float32")},
            {"self_attn.q_proj.weight": ((8, 4), "float32")},
            {"model.embed_tokens.weight": ((64, 4), "float32")},
        ]
        paths = [
            layer_shard_path(str(out), 0),
            layer_shard_path(str(out), 1),
            str(big),
        ]
        return str(out), specs, paths

    def test_the_large_layer_comes_back_exactly_as_disk_source_returns_it(
        self, tmp_path
    ):
        shard_dir, specs, paths = self._hetero(tmp_path)
        shipped = DiskSource(shard_dir, 3, specs, shard_paths=paths)
        ours = AsyncDiskSource(
            shard_dir, 3, specs, shard_paths=paths, read_ahead=2, pin=False
        )
        try:
            for idx, per_layer in enumerate(specs):
                for name in per_layer:
                    theirs = shipped.get(idx, name)
                    mine = ours.get(idx, name)
                    assert mine.shape == theirs.shape, (idx, name)
                    assert torch.equal(mine, theirs), (idx, name)
        finally:
            ours.close()
            shipped.close()

    def test_staging_is_reported_honestly_for_every_distinct_spec(self, tmp_path):
        """Extra host memory is fine; misreporting it is not.

        One slot per distinct spec beyond the decoder's own depth: the embed
        shard is one layer, so depth past 1 there would buy nothing and cost a
        whole vocabulary matrix.
        """
        shard_dir, specs, paths = self._hetero(tmp_path)
        source = AsyncDiskSource(
            shard_dir, 3, specs, shard_paths=paths, read_ahead=2, pin=False
        )
        try:
            decoder = 8 * 4 * 4
            embed = 64 * 4 * 4
            assert source.nbytes == 2 * decoder + embed
            observed = sum(
                dst.numel() * dst.element_size()
                for slot in source._slots
                for dst in slot.values()
            )
            assert source.nbytes == observed, "nbytes disagrees with what was allocated"
        finally:
            source.close()

    def test_disk_bytes_counts_what_the_headers_say(self, tmp_path):
        """Not recomputed from the spec through a fourth dtype-size table."""
        shard_dir, specs, paths = self._hetero(tmp_path)
        source = AsyncDiskSource(
            shard_dir, 3, specs, shard_paths=paths, read_ahead=2, pin=False
        )
        try:
            assert source.disk_bytes == 2 * (8 * 4 * 4) + 64 * 4 * 4
        finally:
            source.close()


# ==========================================================================
# read_ahead must actually read ahead
# ==========================================================================
def _settle(source, timeout: float = 10.0) -> None:
    """Wait until the reader has nothing queued and nothing in flight.

    Depth is a property of the pipeline at rest. Sampling while the reader is
    mid-read would measure this machine's timing, not the source's design.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not source._queue and source._in_flight is None:
            return
        time.sleep(0.002)
    raise AssertionError("the reader never went idle")


def _deep_shards(tmp_path: Path, n_layers: int) -> str:
    from soup_cli.utils.layer_shard import layer_shard_path

    out = tmp_path / "deep"
    out.mkdir()
    for idx in range(n_layers):
        save_file(
            {"w": torch.full((256,), idx % 251, dtype=torch.uint8)},
            layer_shard_path(str(out), idx),
        )
    return str(out)


class TestReadAheadActuallyReadsAhead:
    """``read_ahead`` is a DEPTH, and depth has to be measured, not declared.

    The parent commit armed exactly one target — ``idx + direction`` — so the
    reader was at most ONE layer ahead of the consumer whatever the setting
    said, and ``read_ahead=8`` charged eight layers of pinned host memory for a
    one-deep pipeline. Every test in this file passed. That is what makes a
    setting that does not do what it says invisible (#748), and it matters most
    exactly where this source is used: on a cold disk the read is roughly 12x
    the compute, so one layer of lookahead can only hide one layer's read.

    The reachable depth is ``read_ahead - 1``, not ``read_ahead``: one slot is
    always the one the consumer is holding.
    """

    N_LAYERS = 16

    @pytest.mark.parametrize("read_ahead", [1, 2, 4, 8])
    def test_forward_depth_scales_with_the_setting(self, tmp_path, read_ahead):
        shard_dir = _deep_shards(tmp_path, self.N_LAYERS)
        spec = RamSource.layer_specs_from_shards(shard_dir, self.N_LAYERS)
        source = AsyncDiskSource(
            shard_dir, self.N_LAYERS, spec, read_ahead=read_ahead, pin=False
        )
        try:
            deepest = 0
            for idx in range(self.N_LAYERS - read_ahead):
                source.get(idx, "w")
                _settle(source)
                ahead = sum(1 for layer in source._slot_of if layer > idx)
                deepest = max(deepest, ahead)
            assert deepest == read_ahead - 1, (
                f"read_ahead={read_ahead} staged at most {deepest} layers ahead of "
                f"demand, expected {read_ahead - 1}. The setting charges "
                f"{read_ahead} layers of host memory for the depth it promises."
            )
        finally:
            source.close()

    def test_depth_is_a_window_that_slides_not_a_one_off_burst(self, tmp_path):
        """Depth has to be SUSTAINED, or the pipeline drains after one step."""
        shard_dir = _deep_shards(tmp_path, self.N_LAYERS)
        spec = RamSource.layer_specs_from_shards(shard_dir, self.N_LAYERS)
        source = AsyncDiskSource(shard_dir, self.N_LAYERS, spec, read_ahead=4, pin=False)
        try:
            for idx in range(self.N_LAYERS - 4):
                source.get(idx, "w")
                _settle(source)
                staged = sorted(layer for layer in source._slot_of if layer > idx)
                assert staged == [idx + 1, idx + 2, idx + 3], (
                    f"at layer {idx} the lookahead window was {staged}, not the "
                    f"three consecutive layers the consumer is about to ask for"
                )
        finally:
            source.close()


class TestTheDirectionIsFollowedNotAssumed:
    """The direction half had no test: reverting ``_note_direction`` to
    always-forward passed all 21. On the backward recompute an always-forward
    plan targets layers the consumer has just been through, which are still
    resident, so it queues nothing and the pipeline runs dry exactly half the
    time.
    """

    N_LAYERS = 16

    def test_the_backward_walk_is_prefetched_as_deeply_as_the_forward_one(
        self, tmp_path
    ):
        shard_dir = _deep_shards(tmp_path, self.N_LAYERS)
        spec = RamSource.layer_specs_from_shards(shard_dir, self.N_LAYERS)
        source = AsyncDiskSource(shard_dir, self.N_LAYERS, spec, read_ahead=4, pin=False)
        try:
            deepest = 0
            for idx in range(self.N_LAYERS - 1, 3, -1):
                source.get(idx, "w")
                _settle(source)
                behind = sum(1 for layer in source._slot_of if layer < idx)
                deepest = max(deepest, behind)
            assert deepest == 3, (
                f"walking DOWN, the deepest lookahead was {deepest} layers, expected "
                f"3. The reader is prefetching in the direction the consumer came "
                f"from, not the one it is going."
            )
        finally:
            source.close()

    def test_every_backward_target_is_staged_before_it_is_asked_for(self, tmp_path):
        """The reviewer's shape: after get(idx) on the way down, is idx-1 there?"""
        shard_dir = _deep_shards(tmp_path, 8)
        spec = RamSource.layer_specs_from_shards(shard_dir, 8)
        source = AsyncDiskSource(shard_dir, 8, spec, read_ahead=2, pin=False)
        try:
            for idx in range(8):
                source.get(idx, "w")
            missed = []
            for idx in range(7, 0, -1):
                source.get(idx, "w")
                _settle(source)
                if (idx - 1) not in source._slot_of:
                    missed.append(idx - 1)
            assert not missed, (
                f"layers {missed} were not staged before the backward walk reached "
                f"them — each one is a read the consumer had to wait on"
            )
        finally:
            source.close()


class TestPinningRefusesPageableMemory:
    """The mirror of RamSource's guard (tests/test_qwen35_streaming.py) — the
    branch existed with no test, so a box quietly handing back pageable memory
    would have reported the fast path while paying the ~97% -> ~79% cost.
    """

    def test_a_pageable_allocation_is_refused_not_reported_as_pinned(
        self, tmp_path, monkeypatch
    ):
        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        real_empty = torch.empty

        def _pageable_empty(*args, **kwargs):
            allocation = dict(kwargs)
            allocation["pin_memory"] = False
            return real_empty(*args, **allocation)

        monkeypatch.setattr(torch, "empty", _pageable_empty)
        with pytest.raises(RuntimeError, match="returned pageable memory"):
            AsyncDiskSource(shard_dir, N_LAYERS, spec, pin=True)


class TestDepthSurvivesTheTurnaround:
    """Depth measured on a fresh one-way walk is not the depth training gets.

    Training walks forward, backward, forward, forever. The single-direction
    tests above pass while delivered depth decays across the turnaround: FIFO
    rotation is only in phase with the walk while the walk keeps going the same
    way, so after a reversal it starts evicting layers still inside the
    lookahead window -- the very next layer wanted, in the traced case. Measured
    over 4 sweeps of 32 layers before the eviction policy was fixed: 2 of 3
    sustained at read_ahead=4, and 5 of 7 at 8 with demand misses in both
    directions.

    This drives the PRODUCTION access pattern -- ``get`` immediately followed by
    ``release``, which is what all four ``_release_source`` call sites do.
    """

    N_LAYERS = 32
    SWEEPS = 4

    @staticmethod
    def _sweep(source, read_ahead, n_layers, sweeps):
        """Walk up and down, returning (sustained depth, demand misses)."""
        depths = []
        misses = 0
        for sweep in range(sweeps):
            descending = sweep % 2 == 1
            walk = range(n_layers - 1, -1, -1) if descending else range(n_layers)
            for idx in walk:
                staged_before_demand = idx in source._slot_of
                source.get(idx, "w")
                source.release(idx, None)
                _settle(source)
                if descending:
                    depth = sum(1 for layer in source._slot_of if layer < idx)
                    room = idx >= read_ahead - 1
                else:
                    depth = sum(1 for layer in source._slot_of if layer > idx)
                    room = idx + read_ahead - 1 < n_layers
                # The first sweep is cold, and the ends of a walk simply run
                # out of layers to stage -- neither is the steady state.
                if sweep >= 1 and room:
                    depths.append(depth)
                    if not staged_before_demand:
                        misses += 1
        return min(depths), misses

    @pytest.mark.parametrize("read_ahead", [2, 4, 8])
    def test_full_depth_is_sustained_across_repeated_reversals(
        self, tmp_path, read_ahead
    ):
        shard_dir = _deep_shards(tmp_path, self.N_LAYERS)
        spec = RamSource.layer_specs_from_shards(shard_dir, self.N_LAYERS)
        source = AsyncDiskSource(
            shard_dir, self.N_LAYERS, spec, read_ahead=read_ahead, pin=False
        )
        try:
            sustained, misses = self._sweep(
                source, read_ahead, self.N_LAYERS, self.SWEEPS
            )
            assert sustained == read_ahead - 1, (
                f"read_ahead={read_ahead} peaks at the depth it promises on a "
                f"one-way walk but only SUSTAINS {sustained} of {read_ahead - 1} "
                f"across {self.SWEEPS} reversals. The eviction policy is dropping "
                f"layers still inside the lookahead window."
            )
            assert misses == 0, (
                f"{misses} layers were demanded before they were staged, across "
                f"{self.SWEEPS} sweeps -- each one is a read the consumer waited on "
                f"that the configured depth had already paid for"
            )
        finally:
            source.close()

    def test_a_layer_inside_the_window_is_not_the_eviction_victim(self, tmp_path):
        """The mechanism, asserted directly rather than through its symptom."""
        shard_dir = _deep_shards(tmp_path, self.N_LAYERS)
        spec = RamSource.layer_specs_from_shards(shard_dir, self.N_LAYERS)
        source = AsyncDiskSource(shard_dir, self.N_LAYERS, spec, read_ahead=4, pin=False)
        try:
            for idx in range(self.N_LAYERS):
                source.get(idx, "w")
                source.release(idx, None)
            evicted_while_wanted = []
            for idx in range(self.N_LAYERS - 1, -1, -1):
                source.get(idx, "w")
                source.release(idx, None)
                _settle(source)
                window = {idx - step for step in range(4) if idx - step >= 0}
                staged = set(source._slot_of)
                missing = sorted(window - staged)
                if missing and idx >= 3:
                    evicted_while_wanted.append((idx, missing))
            assert not evicted_while_wanted, (
                f"walking down, these layers were inside the lookahead window but "
                f"not staged: {evicted_while_wanted[:4]}"
            )
        finally:
            source.close()
