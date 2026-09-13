"""#927 — parse a safetensors header without mapping the file."""

import json
import struct
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from soup_cli.utils.safetensors_reader import TensorRange, read_header  # noqa: E402


def _mixed_shard(tmp_path: Path) -> str:
    """An NF4-shaped shard: packed nibbles are uint8, statistics are float32."""
    path = tmp_path / "layer_000.safetensors"
    save_file(
        {
            "self_attn.q_proj.weight": torch.randint(0, 255, (64, 32), dtype=torch.uint8),
            "self_attn.q_proj.weight::absmax": torch.rand(16, dtype=torch.float32),
            "input_layernorm.weight": torch.rand(64, dtype=torch.bfloat16),
        },
        str(path),
    )
    return str(path)


class TestHeaderMatchesSafetensors:
    def test_names_dtypes_and_shapes_match_safe_open(self, tmp_path):
        path = _mixed_shard(tmp_path)
        ours = read_header(path)
        with safe_open(path, framework="pt") as handle:
            assert set(ours) == set(handle.keys())
            for name in handle.keys():
                sliced = handle.get_slice(name)
                assert ours[name].shape == tuple(int(d) for d in sliced.get_shape())

    def test_dtypes_use_soups_spelling(self, tmp_path):
        ours = read_header(_mixed_shard(tmp_path))
        assert ours["self_attn.q_proj.weight"].dtype == "uint8"
        assert ours["self_attn.q_proj.weight::absmax"].dtype == "float32"
        assert ours["input_layernorm.weight"].dtype == "bfloat16"

    def test_byte_ranges_address_the_real_tensor_bytes(self, tmp_path):
        path = _mixed_shard(tmp_path)
        ours = read_header(path)
        with safe_open(path, framework="pt") as handle:
            expected = handle.get_tensor("self_attn.q_proj.weight")
        entry = ours["self_attn.q_proj.weight"]
        with open(path, "rb") as fh:
            fh.seek(entry.start)
            raw = fh.read(entry.end - entry.start)
        assert len(raw) == expected.numel() * expected.element_size()
        assert raw == bytes(expected.reshape(-1).view(torch.uint8).numpy())

    def test_metadata_key_is_not_a_tensor(self, tmp_path):
        path = tmp_path / "meta.safetensors"
        save_file({"w": torch.zeros(4)}, str(path), metadata={"format": "pt"})
        assert set(read_header(str(path))) == {"w"}


class TestHeaderRefusesRatherThanGuesses:
    def test_a_truncated_file_is_refused(self, tmp_path):
        path = tmp_path / "short.safetensors"
        path.write_bytes(struct.pack("<Q", 4096) + b"{}")
        with pytest.raises(ValueError, match="header is truncated"):
            read_header(str(path))

    def test_an_oversized_header_is_refused(self, tmp_path):
        from soup_cli.utils.safetensors_reader import _MAX_HEADER_BYTES

        path = tmp_path / "huge.safetensors"
        path.write_bytes(struct.pack("<Q", _MAX_HEADER_BYTES + 1))
        with pytest.raises(ValueError, match="header claims"):
            read_header(str(path))

    def test_an_unsupported_dtype_is_refused_by_name(self, tmp_path):
        path = tmp_path / "odd.safetensors"
        body = json.dumps(
            {"w": {"dtype": "F8_E4M3", "shape": [2], "data_offsets": [0, 2]}}
        ).encode()
        path.write_bytes(struct.pack("<Q", len(body)) + body + b"\x00\x00")
        with pytest.raises(ValueError, match="F8_E4M3"):
            read_header(str(path))

    def test_offsets_outside_the_file_are_refused(self, tmp_path):
        path = tmp_path / "bad.safetensors"
        body = json.dumps(
            {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}
        ).encode()
        path.write_bytes(struct.pack("<Q", len(body)) + body + b"\x00" * 4)
        with pytest.raises(ValueError, match="past the end of the file"):
            read_header(str(path))

    def test_a_shape_that_disagrees_with_its_byte_range_is_refused(self, tmp_path):
        path = tmp_path / "mismatch.safetensors"
        body = json.dumps(
            {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 8]}}
        ).encode()
        path.write_bytes(struct.pack("<Q", len(body)) + body + b"\x00" * 8)
        with pytest.raises(ValueError, match="byte range"):
            read_header(str(path))

    def test_a_negative_start_offset_is_refused(self, tmp_path):
        """A negative data_offsets[0] would otherwise address bytes inside the
        JSON header itself rather than tensor data — reproduced upstream as
        TensorRange(start=66, end=70) reading back the tail of the header
        text with no exception raised."""
        path = tmp_path / "negative_start.safetensors"
        body = json.dumps(
            {"w": {"dtype": "F32", "shape": [1], "data_offsets": [-4, 0]}}
        ).encode()
        path.write_bytes(struct.pack("<Q", len(body)) + body)
        with pytest.raises(ValueError, match="tensor-data region"):
            read_header(str(path))


def test_tensor_range_is_frozen():
    entry = TensorRange(name="w", dtype="float32", shape=(2,), start=0, end=8)
    with pytest.raises(Exception):
        entry.start = 1


class TestReadIntoFillsPreallocatedTensors:
    def _entry_and_expected(self, tmp_path, name):
        from safetensors import safe_open

        path = _mixed_shard(tmp_path)
        with safe_open(path, framework="pt") as handle:
            expected = handle.get_tensor(name).clone()
        return path, read_header(path)[name], expected

    @pytest.mark.parametrize(
        "name",
        [
            "self_attn.q_proj.weight",
            "self_attn.q_proj.weight::absmax",
            "input_layernorm.weight",
        ],
    )
    def test_bytes_match_safetensors_for_every_dtype(self, tmp_path, name):
        from soup_cli.utils.safetensors_reader import read_into

        path, entry, expected = self._entry_and_expected(tmp_path, name)
        dst = torch.empty(entry.shape, dtype=getattr(torch, entry.dtype), device="cpu")
        with open(path, "rb") as handle:
            read_into(handle, entry, dst)
        assert dst.dtype == expected.dtype
        assert torch.equal(dst.view(torch.uint8), expected.view(torch.uint8))

    def test_a_tensor_of_the_wrong_size_is_refused(self, tmp_path):
        from soup_cli.utils.safetensors_reader import read_into

        path, entry, _ = self._entry_and_expected(tmp_path, "input_layernorm.weight")
        dst = torch.empty((entry.shape[0] + 1,), dtype=torch.bfloat16)
        with open(path, "rb") as handle:
            with pytest.raises(ValueError, match="destination holds"):
                read_into(handle, entry, dst)

    def test_a_non_contiguous_destination_is_refused(self, tmp_path):
        from soup_cli.utils.safetensors_reader import read_into

        path, entry, _ = self._entry_and_expected(tmp_path, "self_attn.q_proj.weight")
        dst = torch.empty((entry.shape[0], entry.shape[1] * 2), dtype=torch.uint8)[:, ::2]
        assert not dst.is_contiguous()
        with open(path, "rb") as handle:
            with pytest.raises(ValueError, match="contiguous"):
                read_into(handle, entry, dst)

    def test_a_truncated_file_raises_rather_than_leaving_stale_bytes(self, tmp_path):
        from soup_cli.utils.safetensors_reader import read_into

        path, entry, _ = self._entry_and_expected(tmp_path, "self_attn.q_proj.weight")
        truncated = tmp_path / "cut.safetensors"
        truncated.write_bytes(Path(path).read_bytes()[: entry.start + 8])
        dst = torch.empty(entry.shape, dtype=torch.uint8)
        with open(truncated, "rb") as handle:
            with pytest.raises(OSError, match="short read"):
                read_into(handle, entry, dst)
