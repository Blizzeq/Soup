"""#927 — training.stream_read_ahead, mirroring stream_buffers."""

import pytest

from soup_cli.config.loader import load_config_from_string
from soup_cli.utils.async_disk_source import DEFAULT_STREAM_READ_AHEAD

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


class TestItIsRefusedWhereItCannotTakeEffect:
    """`stream_source: ram` INSISTS on the RAM tier — it never falls back to
    disk (stream_setup: 'ram' insists, 'disk' forces, 'auto' falls back). The
    read-ahead reader belongs to the disk tier, so a non-default depth beside
    `ram` is a setting that validates, documents and does nothing: #748's class,
    which this very field was caught by.
    """

    def _yaml(self, *, source: str, depth: str = "") -> str:
        return _BASE + f"  stream_source: {source}\n" + depth

    def test_a_non_default_depth_beside_ram_is_refused(self):
        with pytest.raises(ValueError) as excinfo:
            load_config_from_string(
                self._yaml(source="ram", depth="  stream_read_ahead: 4\n")
            )
        message = str(excinfo.value)
        # BOTH names: the user set one field and is being refused because of
        # the other, so a message naming only one leaves them guessing.
        assert "stream_read_ahead" in message, message
        assert "stream_source" in message, message
        assert "4" in message, message

    def test_the_default_is_accepted_beside_ram(self):
        """A default is not a decision. Refusing it would make `stream_source:
        ram` unusable with any config that never mentions the depth at all."""
        config = load_config_from_string(self._yaml(source="ram"))
        assert config.training.stream_source == "ram"
        assert config.training.stream_read_ahead == DEFAULT_STREAM_READ_AHEAD

    @pytest.mark.parametrize("source", ["auto", "disk"])
    @pytest.mark.parametrize("depth", [1, 4, 8])
    def test_every_depth_is_accepted_on_the_tiers_that_can_use_it(self, source, depth):
        """The controls. Without them a validator that refused a non-default
        depth outright — or on every tier — would pass the refusal test."""
        config = load_config_from_string(
            self._yaml(source=source, depth=f"  stream_read_ahead: {depth}\n")
        )
        assert config.training.stream_read_ahead == depth
        assert config.training.stream_source == source
