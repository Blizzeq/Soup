"""#927 — training.stream_read_ahead, mirroring stream_buffers."""

import pytest

from soup_cli.config.loader import load_config_from_string

_BASE = """
base: meta-llama/Llama-3.1-8B
task: sft
data:
  train: data.jsonl
training:
  batch_size: 1
  quantization: 4bit
  stream_layers: true
  lora:
    r: 8
"""


def test_default_is_two():
    config = load_config_from_string(_BASE)
    assert config.training.stream_read_ahead == 2


@pytest.mark.parametrize("value", [1, 4, 8])
def test_accepts_the_documented_range(value):
    config = load_config_from_string(_BASE + f"  stream_read_ahead: {value}\n")
    assert config.training.stream_read_ahead == value


@pytest.mark.parametrize("value", [0, 9, -1])
def test_refuses_out_of_range_by_name(value):
    with pytest.raises(ValueError, match="stream_read_ahead"):
        load_config_from_string(_BASE + f"  stream_read_ahead: {value}\n")


def test_refuses_bool_as_int():
    with pytest.raises(ValueError, match="must be an int, not bool"):
        load_config_from_string(_BASE + "  stream_read_ahead: true\n")


def test_is_footgun_rejected_while_streaming_is_off():
    off = _BASE.replace("stream_layers: true", "stream_layers: false")
    with pytest.raises(ValueError, match="stream_read_ahead"):
        load_config_from_string(off + "  stream_read_ahead: 4\n")


def test_the_schema_bound_and_the_runtime_bound_are_the_same_object():
    """A second copy of the bound would let the message and the check disagree."""
    from soup_cli.utils.async_disk_source import (
        MAX_STREAM_READ_AHEAD,
        MIN_STREAM_READ_AHEAD,
    )
    from soup_cli.utils.layer_stream import (
        MAX_STREAM_READ_AHEAD as LS_MAX,
    )
    from soup_cli.utils.layer_stream import (
        MIN_STREAM_READ_AHEAD as LS_MIN,
    )

    assert (LS_MIN, LS_MAX) == (MIN_STREAM_READ_AHEAD, MAX_STREAM_READ_AHEAD)
