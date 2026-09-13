"""#927 — the disk tier reads on a background thread, not on the compute thread."""

import threading
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
