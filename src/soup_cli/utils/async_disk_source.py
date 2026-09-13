"""Tier 2, asynchronous: the base stays on NVMe and a background thread reads it.

The shipped ``DiskSource`` calls ``safe_open(...).get_tensor`` from
``LayerBufferPool.load_async``, which is only "async" about the copy to the GPU:
the read happens on the COMPUTE thread before the copy is enqueued, so the
Python thread blocks, the GPU starves, and the starvation shows up as wall time
rather than as a stall. Measured cold on a 36 GB 70B-shaped store: 0.57 GB/s
average and 22 MB/s at worst, from a drive that reads 3.5+ GB/s.

This source reads ahead on its own thread into pre-allocated pinned host
buffers, so ``get`` is a handoff. It never memory-maps: a mapping charges
Windows commit for the file's whole size (#926), and holding one per decoder
layer costs ~35 GB of charge for a 70B run.

``get(idx, name)`` keeps the interface ``RamSource`` and ``DiskSource`` share,
so the buffer pool, the prefetcher and the layer wrapper are untouched and the
v0.72.0 correctness gates carry over rather than being re-derived.

NO top-level torch: this module is imported by the trainer path only.
"""

import logging
import math
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from soup_cli.utils.safetensors_reader import TensorRange, read_header, read_into

logger = logging.getLogger(__name__)

MIN_STREAM_READ_AHEAD = 1
MAX_STREAM_READ_AHEAD = 8
DEFAULT_STREAM_READ_AHEAD = 2

_ITEMSIZE = {
    "bfloat16": 2, "float16": 2, "float32": 4, "float64": 8,
    "int8": 1, "int16": 2, "int32": 4, "int64": 8, "uint8": 1, "bool": 1,
}


class AsyncDiskSource:
    """Read layers ahead on a background thread; ``get`` hands over the result."""

    def __init__(
        self,
        shard_dir: str,
        n_layers: int,
        spec: Union[
            Mapping[str, Tuple[Tuple[int, ...], str]],
            Sequence[Mapping[str, Tuple[Tuple[int, ...], str]]],
        ],
        *,
        shard_paths: Optional[Sequence[str]] = None,
        read_ahead: int = DEFAULT_STREAM_READ_AHEAD,
        pin: bool = True,
    ):
        import torch

        from soup_cli.utils.layer_stream_runtime import RamSource

        if isinstance(read_ahead, bool):
            raise ValueError("training.stream_read_ahead must be an int, not bool")
        read_ahead = int(read_ahead)
        if read_ahead < MIN_STREAM_READ_AHEAD or read_ahead > MAX_STREAM_READ_AHEAD:
            raise ValueError(
                f"training.stream_read_ahead must be between {MIN_STREAM_READ_AHEAD} "
                f"and {MAX_STREAM_READ_AHEAD}; got {read_ahead}. Each level costs one "
                f"layer of pinned host memory."
            )

        self._layer_specs = RamSource._normalize_layer_specs(spec, n_layers)
        self._paths = RamSource._normalize_shard_paths(shard_dir, n_layers, shard_paths)
        self.n_layers = int(n_layers)
        self.read_ahead = read_ahead
        self.pinned = bool(pin)

        # Headers once, up front: a shard that disagrees with the index would
        # otherwise read the right byte count from the wrong offsets and train
        # on garbage with no error.
        self._ranges: List[Dict[str, TensorRange]] = []
        for idx in range(self.n_layers):
            header = read_header(self._paths[idx])
            for name, (shape, dtype) in self._layer_specs[idx].items():
                entry = header.get(name)
                if entry is None:
                    raise ValueError(
                        f"{self._paths[idx]}: layer {idx} is missing tensor {name!r}"
                    )
                if entry.shape != tuple(shape) or entry.dtype != dtype:
                    raise ValueError(
                        f"{self._paths[idx]}: tensor {name!r} disagrees with the "
                        f"index — shard has {entry.shape} of {entry.dtype}, the "
                        f"index expects {tuple(shape)} of {dtype}"
                    )
            self._ranges.append(header)

        self.disk_bytes = sum(
            math.prod(shape) * _ITEMSIZE[dtype]
            for per_layer in self._layer_specs
            for shape, dtype in per_layer.values()
        )

        # `read_ahead` staging slots, each one layer's worth, allocated once.
        self._slots: List[Dict[str, Any]] = []
        self.nbytes = 0
        for _ in range(read_ahead):
            slot: Dict[str, Any] = {}
            for name, (shape, dtype) in self._layer_specs[0].items():
                dst = torch.empty(
                    tuple(shape),
                    dtype=getattr(torch, dtype),
                    device="cpu",
                    pin_memory=self.pinned,
                )
                if dst.device.type != "cpu":
                    raise RuntimeError(
                        "layer streaming's async disk source requested a CPU "
                        f"tensor, but torch returned {dst.device}."
                    )
                slot[name] = dst
                self.nbytes += dst.numel() * dst.element_size()
            self._slots.append(slot)

        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)
        self._wanted: Optional[int] = 0
        self._slot_of: Dict[int, int] = {}
        self._in_flight: Optional[int] = None
        self._next_slot = 0
        self._error: Optional[BaseException] = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="soup-layer-reader", daemon=True
        )
        self._thread.start()

    # -- the reader ------------------------------------------------------
    def _run(self) -> None:
        try:
            while True:
                with self._ready:
                    while self._wanted is None and not self._closed:
                        self._ready.wait()
                    if self._closed:
                        return
                    idx = self._wanted
                    self._wanted = None
                    if idx in self._slot_of:
                        continue
                    slot_index = self._next_slot
                    self._next_slot = (self._next_slot + 1) % self.read_ahead
                    evicted = [
                        layer for layer, s in self._slot_of.items() if s == slot_index
                    ]
                    for layer in evicted:
                        del self._slot_of[layer]
                    self._in_flight = idx
                slot = self._slots[slot_index]
                with open(self._paths[idx], "rb") as handle:
                    for name, dst in slot.items():
                        read_into(handle, self._ranges[idx][name], dst)
                with self._ready:
                    self._slot_of[idx] = slot_index
                    self._in_flight = None
                    self._ready.notify_all()
        except BaseException as exc:  # noqa: BLE001 — handed to the consumer
            self._fail(exc)

    def _fail(self, exc: BaseException) -> None:
        """Record a reader failure and wake everyone waiting on it."""
        with self._ready:
            self._error = exc
            self._in_flight = None
            self._ready.notify_all()

    # -- the interface ---------------------------------------------------
    def get(self, idx: int, name: str):
        with self._ready:
            while True:
                if self._error is not None:
                    raise self._error
                if self._closed:
                    raise RuntimeError("layer-stream disk source is closed")
                slot_index = self._slot_of.get(idx)
                if slot_index is not None:
                    tensor = self._slots[slot_index][name]
                    nxt = idx + 1
                    # Arming the next read is only safe with a SPARE slot. The
                    # demand pattern only ever chases "current + 1", so the
                    # round robin's eviction target for `nxt` is always
                    # `nxt - read_ahead`. With one slot that is `idx` itself —
                    # the tensor this call is about to hand back — so eagerly
                    # wanting `nxt` would arm the reader to overwrite the very
                    # buffer the caller is still holding a live reference to
                    # (measured: a torch.equal against a second, non-evicting
                    # source flips false while the caller still holds `tensor`).
                    # With >= 2 slots the eviction target is strictly earlier
                    # than `idx`, which the caller is done with.
                    if (
                        self.read_ahead > 1
                        and nxt < self.n_layers
                        and nxt not in self._slot_of
                    ):
                        self._wanted = nxt
                        self._ready.notify_all()
                    return tensor
                if self._in_flight != idx and self._wanted != idx:
                    self._wanted = idx
                    self._ready.notify_all()
                self._ready.wait(timeout=30.0)
                if (
                    self._error is None
                    and not self._closed
                    and idx not in self._slot_of
                    and self._in_flight != idx
                    and self._wanted != idx
                ):
                    raise RuntimeError(
                        f"layer-stream reader made no progress on layer {idx} for "
                        f"30 s. Refusing rather than blocking: a training run that "
                        f"stops without an error is worse than one that fails."
                    )

    def close(self) -> None:
        """Stop the reader and release the staging buffers. Idempotent."""
        with self._ready:
            if self._closed:
                return
            self._closed = True
            self._ready.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=10.0)
        self._slots = []
        self._slot_of = {}

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:  # noqa: BLE001 — interpreter teardown
            pass
