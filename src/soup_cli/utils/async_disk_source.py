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
so the prefetcher and the layer wrapper are untouched and the v0.72.0
correctness gates carry over rather than being re-derived. ONE call is added
rather than kept: ``release(idx, event)``. A reusable staging buffer is exactly
what the other two sources do not have — ``RamSource`` holds every layer for the
whole run and ``DiskSource`` returns a freshly allocated tensor per call, so
neither can be recycled underneath an in-flight copy. This source can, and out
of PINNED host memory ``dst.copy_(..., non_blocking=True)`` is still draining
when ``load_async`` returns while ``pool.wait`` is a GPU-side ``wait_event``
that does not block the Python thread at all. Measured through the real pool
before ``release`` existed: 7 of 8 layers reached the device holding another
layer's weights at ``read_ahead=1``, 6 of 8 at the default 2, 4 of 8 at 4.

NO top-level torch: this module is imported by the trainer path only.
"""

import logging
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from soup_cli.utils.safetensors_reader import TensorRange, read_header, read_into

logger = logging.getLogger(__name__)

MIN_STREAM_READ_AHEAD = 1
MAX_STREAM_READ_AHEAD = 8
DEFAULT_STREAM_READ_AHEAD = 2


def _spec_key(layer_spec: Mapping[str, Tuple[Tuple[int, ...], str]]) -> tuple:
    """A hashable identity for one layer's tensor names, shapes and dtypes.

    Two layers share staging only if they agree on all three: a buffer sized for
    a decoder layer cannot hold a vocabulary matrix, and a buffer keyed on
    ``self_attn.q_proj.weight`` cannot answer ``model.embed_tokens.weight``.
    """
    return tuple(
        (name, tuple(shape), dtype)
        for name, (shape, dtype) in sorted(layer_spec.items())
    )


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

        # The exact bytes that will be read, taken from the headers rather than
        # recomputed from the spec: `read_header` already cross-checks each
        # range against shape x itemsize, so a second dtype-size table here
        # could only ever disagree with the one authority. `layer_stream.
        # dtype_bytes` is THE table for everything that still needs one.
        self.disk_bytes = sum(
            self._ranges[idx][name].nbytes
            for idx in range(self.n_layers)
            for name in self._layer_specs[idx]
        )

        # Staging is allocated per DISTINCT layer spec, NOT from layer 0's.
        # `_build_source` hands this source the decoder layers followed by the
        # vocabulary-sized embed / lm_head shards, whose tensor keys are
        # entirely different; sizing every slot from layer 0 made the reader
        # thread raise KeyError on the first forward turnaround of any real
        # model and poisoned the source permanently, where `DiskSource` simply
        # returns the tensor.
        groups: Dict[tuple, int] = {}
        specs_by_group: List[Mapping[str, Tuple[Tuple[int, ...], str]]] = []
        members: List[int] = []
        self._group_of: List[int] = []
        for idx in range(self.n_layers):
            key = _spec_key(self._layer_specs[idx])
            group = groups.get(key)
            if group is None:
                group = len(groups)
                groups[key] = group
                specs_by_group.append(self._layer_specs[idx])
                members.append(0)
            members[group] += 1
            self._group_of.append(group)

        self._slots: List[Dict[str, Any]] = []
        self._group_slots: List[List[int]] = []
        self.nbytes = 0
        for group, layer_spec in enumerate(specs_by_group):
            # Depth beyond the number of layers sharing a spec buys nothing and
            # costs a whole vocabulary matrix of pinned host memory: embed and
            # an untied lm_head are one layer each.
            flat: List[int] = []
            for _ in range(min(read_ahead, members[group])):
                slot: Dict[str, Any] = {}
                for name, (shape, dtype) in layer_spec.items():
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
                    # Mirrors RamSource: a box that hands back pageable memory
                    # would otherwise report pinned=True while silently paying
                    # the ~97% -> ~79% GPU-utilisation cost of a synchronous
                    # host-to-device copy.
                    if self.pinned and not dst.is_pinned():
                        raise RuntimeError(
                            "layer streaming requested pinned CPU RAM, but torch "
                            "returned pageable memory; retry with pin=False."
                        )
                    slot[name] = dst
                    self.nbytes += dst.numel() * dst.element_size()
                flat.append(len(self._slots))
                self._slots.append(slot)
            self._group_slots.append(flat)

        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)
        self._wanted: Optional[int] = 0
        self._slot_of: Dict[int, int] = {}
        self._in_flight: Optional[int] = None
        self._next_slot: List[int] = [0] * len(self._group_slots)
        # Per staging slot: handed to the consumer and not released yet, and
        # the event that says when its copy has drained.
        self._live: List[bool] = [False] * len(self._slots)
        self._drain: List[Any] = [None] * len(self._slots)
        self._last_get: Optional[int] = None
        self._direction = 1
        self._error: Optional[BaseException] = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="soup-layer-reader", daemon=True
        )
        self._thread.start()

    # -- the reader ------------------------------------------------------
    def _claim_slot(self, idx: int) -> Optional[int]:
        """Choose a staging slot to refill with layer ``idx``. Lock held.

        Only slots in ``idx``'s spec group are eligible — a decoder buffer
        cannot hold a vocabulary matrix. A slot the consumer is still holding
        is never chosen: overwriting one is what put another layer's weights on
        the device through the real buffer pool. ``None`` means every slot in
        the group is held, and the caller waits for a release rather than
        picking a victim anyway — a fallback that overwrites a live slot is the
        defect, not a relief valve for it.
        """
        group = self._group_of[idx]
        flat = self._group_slots[group]
        start = self._next_slot[group]
        for offset in range(len(flat)):
            candidate = flat[(start + offset) % len(flat)]
            if self._live[candidate]:
                continue
            self._next_slot[group] = (start + offset + 1) % len(flat)
            for layer in [lay for lay, held in self._slot_of.items() if held == candidate]:
                del self._slot_of[layer]
            return candidate
        return None

    def _hold(self, idx: int) -> None:
        """Mark ``idx``'s slot as in use and implicitly release the rest. Lock held.

        ``release()`` is the precise signal and the real buffer pool sends it.
        A consumer that does not — every caller that predates this source, and
        the byte-identity gate, which calls ``get`` directly — still gets the
        contract this design always had: a reference from ``get`` is valid until
        your next ``get`` for a different layer. Without that, slots handed out
        to a release-unaware consumer would never come back and the reader would
        stall until ``get``'s own timeout fired.

        The implicit release carries NO drain event, which is correct: a
        consumer that never calls ``release`` never enqueued an asynchronous
        copy out of this buffer either.
        """
        keep = self._slot_of.get(idx)
        freed = False
        for slot in range(len(self._slots)):
            if slot != keep and self._live[slot]:
                self._live[slot] = False
                freed = True
        if keep is not None:
            self._live[keep] = True
        if freed:
            self._ready.notify_all()

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
                    claimed = self._claim_slot(idx)
                    if claimed is None:
                        # Every slot in this group is still handed out. Put the
                        # request back and wait for a release: the alternative
                        # is overwriting a buffer the consumer is reading, which
                        # is the whole defect. If no release ever comes, `get`
                        # surfaces it as its own loud timeout rather than a
                        # silently wrong weight reaching the device.
                        self._wanted = idx
                        self._ready.wait(timeout=1.0)
                        continue
                    slot_index = claimed
                    draining = self._drain[slot_index]
                    self._drain[slot_index] = None
                    self._in_flight = idx
                # OUTSIDE the lock: the compute thread must be able to call
                # get() while this waits. The event was recorded on a stream
                # that already waited on the compute stream, so it depends only
                # on work the GPU has been handed — never on this process
                # taking another Python step, which is what makes waiting here
                # safe rather than a deadlock.
                if draining is not None:
                    draining.synchronize()
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

    def _note_direction(self, idx: int) -> None:
        """Track which way the consumer is walking the stack. Lock held.

        ``StreamPrefetcher`` walks 0..L-1 on the forward pass and L-1..0 on the
        backward recompute, so arming ``idx + 1`` unconditionally spent the
        whole backward half fetching a layer the consumer had already passed —
        wasted I/O that also evicted a slot the next ``get`` wanted. A repeat of
        the same index carries no direction information and leaves it alone.

        A fetch from a DIFFERENT spec group is not a step along this walk: the
        tail prefetch of ``model.embed_tokens.weight`` sits at an index above
        every decoder layer, so letting it speak would read as a forward step at
        exactly the turnaround where the direction has just flipped. It is
        ignored entirely — including for ``_last_get``, so the decoder step
        after it still compares against the last decoder step.
        """
        last = self._last_get
        if last is not None and self._group_of[idx] != self._group_of[last]:
            return
        self._last_get = idx
        if last is not None and idx != last:
            self._direction = 1 if idx > last else -1

    # -- the interface ---------------------------------------------------
    def get(self, idx: int, name: str):
        """Hand back layer ``idx``'s ``name``, staged in a REUSABLE host buffer.

        This is a borrow, not a copy — which is the point, and which is the one
        way this source differs from ``RamSource`` (holds everything forever)
        and ``DiskSource`` (allocates per call). Two rules follow:

        * The reference is valid until your next ``get`` for a different layer.
        * If you enqueue an ASYNCHRONOUS copy out of it, you must say so with
          ``release(idx, event)`` — otherwise the next ``get`` is taken as
          "done", the reader refills the buffer, and your copy drains from
          whatever it now holds. ``LayerBufferPool`` and ``LargeLayerBufferPool``
          both do this; see ``layer_stream_runtime._release_source``.
        """
        with self._ready:
            while True:
                if self._error is not None:
                    raise self._error
                if self._closed:
                    raise RuntimeError("layer-stream disk source is closed")
                slot_index = self._slot_of.get(idx)
                if slot_index is not None:
                    tensor = self._slots[slot_index][name]
                    # Held from here until release() says the consumer's copy
                    # has drained. Without this the window between handing the
                    # reference out and the copy being enqueued is enough for
                    # the reader to overwrite it.
                    self._hold(idx)
                    self._note_direction(idx)
                    nxt = idx + self._direction
                    # Arming the next read only buys overlap with a SPARE slot.
                    # At depth 1 the only slot is the one being handed back, so
                    # the reader would claim nothing and wait for its release —
                    # pure overhead. (Before the live-slot rule below, it did
                    # something worse: it armed the reader to overwrite the very
                    # buffer this call was returning, which is how a read_ahead=1
                    # source handed back another layer's bytes.)
                    if (
                        self.read_ahead > 1
                        and 0 <= nxt < self.n_layers
                        and nxt not in self._slot_of
                    ):
                        self._wanted = nxt
                        self._ready.notify_all()
                    return tensor
                # Asking for a layer that is not resident ends any implicit hold
                # on the others, so the reader always has a slot to claim. This
                # is what keeps a release-unaware consumer from deadlocking a
                # reader that now refuses to overwrite a live slot.
                self._hold(idx)
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

    def release(self, idx: int, event: Any = None) -> None:
        """Say the consumer is done reading layer ``idx`` out of its staging slot.

        ``LayerBufferPool.load_async`` enqueues ``dst.copy_(..., non_blocking=
        True)`` on a side stream out of PINNED host memory, so the copy is still
        draining when the call returns, and ``pool.wait`` is a GPU-side
        ``wait_event`` that never blocks the Python thread. Recycling the
        staging buffer at that point rewrites the bytes the copy is reading:
        measured through the real pool, 6 of 8 layers reached the device holding
        another layer's weights at the default depth.

        ``event`` is a CUDA event recorded after those copies — the reader waits
        on it before reusing the slot. ``None`` means the copy was already
        synchronous (no side stream, or a non-CUDA device) and the slot is free
        immediately. A layer that is no longer resident is ignored: it can only
        have been evicted, which requires having been released already.
        """
        with self._ready:
            slot_index = self._slot_of.get(idx)
            if slot_index is None:
                return
            self._drain[slot_index] = event
            self._live[slot_index] = False
            # The reader may be parked because every slot in this group was
            # held; this is the event it was waiting for.
            self._ready.notify_all()

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
