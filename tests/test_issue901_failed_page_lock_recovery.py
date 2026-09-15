"""#901 — a failed page-lock left a stale CUDA error that killed the next kernel launch.

The report: Qwen2.5-14B, ``stream_layers: true``, NF4, on an 8 GB card with 32 GB
of RAM. The pre-flight predicted a 3.39 GB peak against 7.41 GB free, the run
printed "could not page-lock the base ... falling back to a PAGEABLE RAM store",
and then died with ``AcceleratorError: CUDA error: out of memory`` inside
``SFTTrainer.__init__`` — or, with ``training.stream_vram_probe`` on, inside the
probe's first forward. Lowering ``max_length`` changed nothing.

Reproduced on the dev box (RTX 5070 Laptop 8 GB, Windows 11, torch 2.14.0+cu130)
with a synthetic Qwen2.5-14B-shaped checkpoint, then reduced to pure torch:

* ``torch.empty(N, pin_memory=True)`` for N past what ``cuMemHostAlloc`` will
  give raises ``AcceleratorError("CUDA error: out of memory")`` in 0.1 s.
* After that, ``cudaMalloc``, ``synchronize`` and a host-to-device ``memcpy``
  all succeed — but the FIRST KERNEL LAUNCH raises the same "out of memory"
  with 7.3 GB of VRAM free, and the second launch works. The failed host
  allocation leaves the runtime's per-thread "last error" set, nothing on the
  fallback path clears it, and the launch check of the next kernel reads it.
  In the report that next kernel was ``param.data.to(torch.bfloat16)``; in the
  probe it was the forward.

So the pageable fallback was dead on arrival: it announced itself and then
handed the run a poisoned context. Memory was never the problem.

There is no ``cudaGetLastError`` binding in ``torch.cuda.cudart()``, so the
drain launches one trivial kernel and lets its launch check consume the stale
error, then launches a second to prove the context is healthy. The failed
attempt's page-locked blocks also stay in torch's caching host allocator until
released, so the recovery empties that cache too: on the 14B store that is
gigabytes of dead pinned memory otherwise held for the whole run.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def _cuda() -> bool:
    import torch

    return torch.cuda.is_available()


requires_cuda = pytest.mark.skipif(not _cuda(), reason="needs a CUDA device")

#: Far beyond any box's RAM, so ``cuMemHostAlloc`` refuses it at once without
#: touching memory — the fastest way to leave the stale error behind.
_IMPOSSIBLE_PIN_BYTES = 2**40


class _Console:
    def __init__(self):
        self.printed = []

    def print(self, msg):
        self.printed.append(str(msg))


class TestDrainStaleCudaError:
    """The drain: consume a stale error with one launch, prove health with a second."""

    @staticmethod
    def _launch_with(outcomes):
        calls = []

        def launch():
            calls.append(len(calls))
            outcome = outcomes.pop(0)
            if outcome is not None:
                raise outcome

        return launch, calls

    def test_a_stale_out_of_memory_is_consumed_and_the_second_launch_proves_the_context(self):
        from soup_cli.utils.layer_stream_runtime import drain_stale_cuda_error

        launch, calls = self._launch_with([RuntimeError("CUDA error: out of memory"), None])
        assert drain_stale_cuda_error("cuda", launch=launch) is True
        assert len(calls) == 2

    def test_a_healthy_context_launches_once_and_reports_nothing_drained(self):
        from soup_cli.utils.layer_stream_runtime import drain_stale_cuda_error

        launch, calls = self._launch_with([None])
        assert drain_stale_cuda_error("cuda", launch=launch) is False
        assert len(calls) == 1

    def test_two_failures_in_a_row_propagate_as_a_genuinely_broken_context(self):
        from soup_cli.utils.layer_stream_runtime import drain_stale_cuda_error

        launch, calls = self._launch_with(
            [RuntimeError("CUDA error: out of memory"), RuntimeError("CUDA error: out of memory")]
        )
        with pytest.raises(RuntimeError, match="out of memory"):
            drain_stale_cuda_error("cuda", launch=launch)
        assert len(calls) == 2

    def test_an_error_that_is_not_out_of_memory_is_never_swallowed(self):
        from soup_cli.utils.layer_stream_runtime import drain_stale_cuda_error

        launch, calls = self._launch_with(
            [RuntimeError("CUDA error: an illegal memory access was encountered"), None]
        )
        with pytest.raises(RuntimeError, match="illegal memory access"):
            drain_stale_cuda_error("cuda", launch=launch)
        assert len(calls) == 1

    def test_without_a_cuda_device_there_is_nothing_to_drain(self, monkeypatch):
        """No context, no stale error — and no attempt to launch on a device
        that is not there, which would itself raise on a CPU-only box."""
        import torch

        from soup_cli.utils.layer_stream_runtime import drain_stale_cuda_error

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert drain_stale_cuda_error("cuda") is False


class TestTheFallbackRecoversBeforeBuildingThePageableStore:
    """``_build_source``: the recovery runs BETWEEN the failed pinned constructor
    and the pageable one, on both tiers. After the pageable store is built the
    next CUDA op is the first kernel of the run, so recovering after it would be
    exactly as late as never."""

    def test_ram_tier_recovers_before_the_pageable_store_is_built(self, tmp_path, monkeypatch):
        import soup_cli.utils.layer_stream_runtime as rt
        from soup_cli.utils.layer_shard import shard_checkpoint
        from tests.test_v07200 import _tiny_llama_dir

        weights, _, _ = _tiny_llama_dir(tmp_path)
        shards = str(tmp_path / "shards")
        index = shard_checkpoint(weights, shards, dtype="float32")
        spec = rt.RamSource.spec_from_shard(shards)
        events = []
        real = rt.RamSource

        class _FailsWhenPinned(real):
            def __init__(self, shard_dir, n_layers, spec, *, pin=True):
                events.append(("ramsource", pin))
                if pin:
                    raise RuntimeError("CUDA error: out of memory")
                super().__init__(shard_dir, n_layers, spec, pin=False)

        monkeypatch.setattr(rt, "RamSource", _FailsWhenPinned)
        monkeypatch.setattr(
            rt,
            "recover_from_failed_page_lock",
            lambda **kwargs: events.append("recover") or True,
        )
        console = _Console()
        source, pinned = rt._build_source(shards, index.n_layers, spec, True, console)
        assert events == [("ramsource", True), "recover", ("ramsource", False)]
        assert pinned is False
        assert source.nbytes > 0

    def test_disk_tier_recovers_before_the_pageable_staging_is_built(self, tmp_path, monkeypatch):
        import soup_cli.utils.async_disk_source as ads
        import soup_cli.utils.layer_stream_runtime as rt
        from tests.test_issue971_async_disk_source import N_LAYERS, _shards, _spec

        shard_dir = _shards(tmp_path)
        events = []
        real = ads.AsyncDiskSource

        class _FailsWhenPinned(real):
            def __init__(self, *args, pin=True, **kwargs):
                events.append(("asyncsource", pin))
                if pin:
                    raise RuntimeError("CUDA error: out of memory")
                super().__init__(*args, pin=False, **kwargs)

        monkeypatch.setattr(ads, "AsyncDiskSource", _FailsWhenPinned)
        monkeypatch.setattr(
            rt,
            "recover_from_failed_page_lock",
            lambda **kwargs: events.append("recover") or True,
        )
        console = _Console()
        source, pinned = rt._build_source(
            shard_dir, N_LAYERS, _spec(shard_dir), True, console, "disk", read_ahead=2
        )
        try:
            assert events == [("asyncsource", True), "recover", ("asyncsource", False)]
            assert pinned is False
        finally:
            source.close()

    def test_a_refused_pin_under_stream_pin_true_does_not_build_anything_to_recover_for(
        self, tmp_path, monkeypatch
    ):
        """``require_pin`` raises instead of falling back; the process is refusing,
        so no pageable store is built and no recovery is attempted for one."""
        import soup_cli.utils.layer_stream_runtime as rt
        from soup_cli.utils.layer_shard import shard_checkpoint
        from tests.test_v07200 import _tiny_llama_dir

        weights, _, _ = _tiny_llama_dir(tmp_path)
        shards = str(tmp_path / "shards")
        index = shard_checkpoint(weights, shards, dtype="float32")
        spec = rt.RamSource.spec_from_shard(shards)
        events = []

        class _FailsWhenPinned:
            def __init__(self, shard_dir, n_layers, spec, *, pin=True, **kwargs):
                events.append(("ramsource", pin))
                raise RuntimeError("CUDA error: out of memory")

        monkeypatch.setattr(rt, "RamSource", _FailsWhenPinned)
        monkeypatch.setattr(
            rt,
            "recover_from_failed_page_lock",
            lambda **kwargs: events.append("recover") or True,
        )
        with pytest.raises(RuntimeError, match="stream_pin=true"):
            rt._build_source(shards, index.n_layers, spec, True, _Console(), require_pin=True)
        assert events == [("ramsource", True)]


@requires_cuda
class TestOnRealHardware:
    """The mechanism itself, on a real device. CI has no GPU, so these run on
    dev boxes; the first one is a characterisation of torch/CUDA behaviour and
    is what tells us when the drain stops being necessary."""

    @staticmethod
    def _fail_a_page_lock():
        import torch

        with pytest.raises(RuntimeError):
            torch.empty(_IMPOSSIBLE_PIN_BYTES, dtype=torch.uint8, pin_memory=True)

    @staticmethod
    def _launch():
        import torch

        torch.ones(1, device="cuda")
        torch.cuda.synchronize()

    def test_a_failed_page_lock_poisons_the_next_launch_and_only_the_next(self):
        """The upstream behaviour this fix exists for. If this test ever FAILS
        because the first launch after the failed page-lock succeeds, torch
        or the driver has started clearing the error itself and the drain can
        be retired."""
        self._fail_a_page_lock()
        with pytest.raises(RuntimeError, match="out of memory"):
            self._launch()
        self._launch()

    def test_the_drain_makes_the_first_launch_after_a_failed_page_lock_succeed(self):
        from soup_cli.utils.layer_stream_runtime import drain_stale_cuda_error

        self._fail_a_page_lock()
        assert drain_stale_cuda_error("cuda") is True
        self._launch()

    def test_the_drain_on_a_healthy_context_is_a_no_op(self):
        from soup_cli.utils.layer_stream_runtime import drain_stale_cuda_error

        self._launch()
        assert drain_stale_cuda_error("cuda") is False
        self._launch()

    def test_the_pageable_fallback_leaves_a_usable_context(self, tmp_path, monkeypatch):
        """The whole chain, with a REAL failed page-lock behind the fallback.

        ``torch.empty(pin_memory=True)`` is wrapped so the pinned store's first
        allocation is an impossible one: the genuine driver failure, the
        genuine stale error, the shipped fallback. Without the recovery the
        first kernel after ``_build_source`` returns is the one that dies — in
        the report, inside ``SFTTrainer.__init__``."""
        import torch

        import soup_cli.utils.layer_stream_runtime as rt
        from soup_cli.utils.layer_shard import layer_shard_path, shard_checkpoint
        from tests.test_v07200 import _tiny_llama_dir

        weights, _, _ = _tiny_llama_dir(tmp_path)
        shards = str(tmp_path / "shards")
        index = shard_checkpoint(weights, shards, dtype="float32")
        spec = rt.RamSource.spec_from_shard(shards)
        real_empty = torch.empty

        def _impossible_when_pinned(*args, **kwargs):
            if kwargs.get("pin_memory"):
                return real_empty(_IMPOSSIBLE_PIN_BYTES, dtype=torch.uint8, pin_memory=True)
            return real_empty(*args, **kwargs)

        monkeypatch.setattr(torch, "empty", _impossible_when_pinned)
        console = _Console()
        source, pinned = rt._build_source(shards, index.n_layers, spec, True, console)
        monkeypatch.undo()

        assert pinned is False
        assert any("PAGEABLE" in msg for msg in console.printed)
        # The first kernel launch of the run, where the report died.
        self._launch()
        # And the pageable store is the real one: bit-identical to the shard.
        from safetensors import safe_open

        with safe_open(layer_shard_path(shards, 0), framework="pt") as handle:
            expected = handle.get_tensor("self_attn.q_proj.weight")
        assert torch.equal(source.get(0, "self_attn.q_proj.weight"), expected)

    def test_the_recovery_returns_the_failed_attempts_cached_pinned_blocks(self):
        """A pinned tensor torch frees goes back to its caching host allocator,
        not to the OS. After a partially pinned store is abandoned that cache
        holds gigabytes of page-locked memory for nothing; the recovery
        empties it."""
        import torch

        from soup_cli.utils.layer_stream_runtime import recover_from_failed_page_lock

        def cached_bytes() -> int:
            stats = torch.cuda.host_memory_stats()
            return int(stats["allocated_bytes.current"]) - int(stats["active_bytes.current"])

        held = [torch.empty(2**26, dtype=torch.uint8, pin_memory=True) for _ in range(4)]
        del held
        before = cached_bytes()
        assert before >= 4 * 2**26
        recover_from_failed_page_lock(device="cuda", console=None)
        assert cached_bytes() < before
