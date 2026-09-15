"""#901, the other half — why the 14B store failed to page-lock on a 32 GB box.

``RamSource`` allocated every store tensor with its own ``torch.empty(...,
pin_memory=True)``. torch's caching host allocator rounds each pinned request
up to the next power of two, so the page-locked cost of the store is not its
byte count: measured on the dev box (RTX 5070 Laptop, 31.7 GB RAM, torch
2.14.0+cu130) against the real Qwen2.5-14B NF4 shard cache,

    per-tensor pinning   6.82 GB requested -> 11.83 GB of private commit (1.73x)
    100 x 35.39 MB       3.54 GB requested ->  6.72 GB                   (1.90x)
    one 9 GB block       -> refused outright (it rounds to 16 GiB)
    one 8 GB block       -> fine (2^33 exactly)
    9.93 GB in 5 chunks  -> fine

so the 9.93 GB store in the report asked the driver for ~17 GB of page-locked
memory, which is what "could not page-lock the base" was about, and the old
box's "7.12 GB ceiling" was the same rounding against 16.9 GB of RAM.

The fix packs the pinned store into a few arenas whose sizes ARE powers of two
and carves every tensor as a view, so page-locked demand is the store plus a
bounded tail per arena instead of up to double. The store's bytes, ``get``'s
contract and the pageable path are untouched.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

MiB = 2**20


def _cuda() -> bool:
    import torch

    return torch.cuda.is_available()


requires_cuda = pytest.mark.skipif(not _cuda(), reason="needs a CUDA device")


#: One Qwen2.5-14B NF4 decoder layer, from the shard cache's own headers (bytes):
#: three 35.39 MB packed MLP projections, two 13.11 MB attention projections,
#: two 2.62 MB k/v projections, their absmax / nested-absmax / nested-offset
#: sidecars, two layernorms and the three Qwen2 biases. 30 tensors, 142.1 MB.
_QWEN14B_NF4_LAYER_BYTES = (
    [35_389_440] * 3
    + [13_107_200] * 2
    + [2_621_440] * 2
    + [1_105_920] * 3
    + [409_600] * 2
    + [81_920] * 2
    + [17_280] * 3
    + [6_400] * 2
    + [1_280] * 2
    + [4] * 7
    + [10_240] * 2
    + [10_240, 2_048, 2_048]
)


class TestPlanPinnedArenas:
    """Pure arithmetic: where each tensor lands and what the arenas cost."""

    def test_tensors_pack_into_one_arena_until_it_is_full_then_open_the_next(self):
        from soup_cli.utils.layer_stream_runtime import plan_pinned_arenas

        plan = plan_pinned_arenas([100, 100, 100], arena_bytes=256, align=64)
        # 0..100, then aligned up to 128..228, then 256+100 > 256 opens arena 1.
        assert plan.placements == ((0, 0), (0, 128), (1, 0))
        assert plan.arena_sizes == (256, 128)
        assert plan.requested_bytes == 300
        assert plan.pinned_bytes == 384

    def test_every_arena_is_a_power_of_two_and_no_tensor_straddles_one(self):
        from soup_cli.utils.layer_stream_runtime import plan_pinned_arenas

        sizes = [3_000_000, 700_000, 5, 12_345_678, 1, 0, 999_999] * 9
        plan = plan_pinned_arenas(sizes, arena_bytes=16 * MiB)
        for size in plan.arena_sizes:
            assert size & (size - 1) == 0 and size > 0
        for (arena, offset), size in zip(plan.placements, sizes):
            assert 0 <= offset
            assert offset + size <= plan.arena_sizes[arena]
            assert offset % 256 == 0

    def test_placements_never_overlap_within_an_arena(self):
        from soup_cli.utils.layer_stream_runtime import plan_pinned_arenas

        sizes = [1_000_003, 77, 4_194_304, 300, 2_500_000, 8]
        plan = plan_pinned_arenas(sizes, arena_bytes=8 * MiB)
        spans = sorted(
            (arena, offset, offset + size)
            for (arena, offset), size in zip(plan.placements, sizes)
            if size
        )
        for (a_arena, _a_lo, a_hi), (b_arena, b_lo, _b_hi) in zip(spans, spans[1:]):
            if a_arena == b_arena:
                assert a_hi <= b_lo

    def test_a_tensor_larger_than_the_arena_gets_a_power_of_two_arena_of_its_own(self):
        from soup_cli.utils.layer_stream_runtime import plan_pinned_arenas

        plan = plan_pinned_arenas([600, 10], arena_bytes=256, align=64)
        assert plan.placements[0] == (0, 0)
        assert plan.arena_sizes[0] == 1024

    def test_the_last_arena_is_trimmed_to_what_it_holds(self):
        """A tiny model must not page-lock a whole default arena for a 5 MB store."""
        from soup_cli.utils.layer_stream_runtime import plan_pinned_arenas

        plan = plan_pinned_arenas([5 * MiB], arena_bytes=256 * MiB)
        assert plan.arena_sizes == (8 * MiB,)

    def test_the_14b_store_costs_at_most_ten_percent_over_its_bytes(self):
        """The measurement this fix is for: per-tensor pinning cost 1.73x on
        this exact store. 48 layers x 30 tensors, the shard cache's own sizes."""
        from soup_cli.utils.layer_stream_runtime import plan_pinned_arenas

        sizes = _QWEN14B_NF4_LAYER_BYTES * 48
        plan = plan_pinned_arenas(sizes)
        requested = sum(sizes)
        assert abs(requested - 6_820_000_000) < 30_000_000, requested
        assert plan.pinned_bytes <= 1.10 * requested, plan.pinned_bytes / requested

    def test_an_empty_store_plans_no_arenas(self):
        from soup_cli.utils.layer_stream_runtime import plan_pinned_arenas

        plan = plan_pinned_arenas([])
        assert plan.arena_sizes == ()
        assert plan.placements == ()
        assert plan.pinned_bytes == 0

    @pytest.mark.parametrize(
        "kwargs, needle",
        [
            ({"arena_bytes": 3000}, "power of two"),
            ({"align": 48}, "power of two"),
            ({"arena_bytes": 64, "align": 256}, "align"),
        ],
    )
    def test_a_malformed_geometry_is_refused_by_name(self, kwargs, needle):
        from soup_cli.utils.layer_stream_runtime import plan_pinned_arenas

        with pytest.raises(ValueError, match=needle):
            plan_pinned_arenas([10], **kwargs)

    def test_a_negative_size_is_refused(self):
        from soup_cli.utils.layer_stream_runtime import plan_pinned_arenas

        with pytest.raises(ValueError, match="negative"):
            plan_pinned_arenas([10, -1])


def _shards(tmp_path, n_layers: int = 2) -> tuple:
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import RamSource
    from tests.test_v07200 import _tiny_llama_dir

    weights, _, _ = _tiny_llama_dir(tmp_path, n_layers=n_layers)
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(weights, shards, dtype="float32")
    return shards, index, RamSource.layer_specs_from_shards(shards, index.n_layers)


def _shard_tensor(shards: str, idx: int, name: str):
    from safetensors import safe_open

    from soup_cli.utils.layer_shard import layer_shard_path

    with safe_open(layer_shard_path(shards, idx), framework="pt") as handle:
        return handle.get_tensor(name)


class TestPageableStoreIsUntouched:
    def test_pin_false_allocates_no_arenas_and_keeps_the_bytes(self, tmp_path):
        import torch

        from soup_cli.utils.layer_stream_runtime import RamSource

        shards, index, specs = _shards(tmp_path)
        source = RamSource(shards, index.n_layers, specs, pin=False)
        assert source.pinned is False
        assert source.arena_sizes == ()
        assert source.pinned_bytes == 0
        for idx in range(index.n_layers):
            for name in specs[idx]:
                got = source.get(idx, name)
                assert not got.is_pinned()
                assert torch.equal(got, _shard_tensor(shards, idx, name))


@requires_cuda
class TestPinnedStoreLivesInArenas:
    def test_every_store_tensor_is_a_pinned_view_into_a_power_of_two_arena(self, tmp_path):
        from soup_cli.utils.layer_stream_runtime import RamSource

        shards, index, specs = _shards(tmp_path)
        source = RamSource(shards, index.n_layers, specs, pin=True)
        assert source.pinned is True
        storages = set()
        for idx in range(index.n_layers):
            for name in specs[idx]:
                got = source.get(idx, name)
                assert got.is_pinned()
                assert tuple(got.shape) == tuple(specs[idx][name][0])
                storages.add(got.untyped_storage().data_ptr())
        assert len(storages) == len(source.arena_sizes)
        for size in source.arena_sizes:
            assert size & (size - 1) == 0
        assert source.pinned_bytes == sum(source.arena_sizes)
        assert source.nbytes < source.pinned_bytes

    def test_the_pinned_store_is_bit_identical_to_the_shards(self, tmp_path):
        import torch

        from soup_cli.utils.layer_stream_runtime import RamSource

        shards, index, specs = _shards(tmp_path, n_layers=3)
        source = RamSource(shards, index.n_layers, specs, pin=True)
        for idx in range(index.n_layers):
            for name in specs[idx]:
                assert torch.equal(source.get(idx, name), _shard_tensor(shards, idx, name)), (
                    idx,
                    name,
                )

    def test_the_host_allocator_is_charged_exactly_the_arena_sizes(self, tmp_path):
        """``active_bytes.current`` is what torch's caching host allocator holds
        live for this process, at the ROUNDED size: measured, a 100 000-byte
        request moves it by 131 072 and a 35 389 440-byte one by 67 108 864.
        With arenas it moves by exactly the arenas' power-of-two sizes, so
        ``pinned_bytes`` is the figure the box really pays. (The saving itself
        is a property of stores that span many arenas — the 14B arithmetic
        above; a fixture this small fits one arena and gains nothing.)"""
        import torch

        from soup_cli.utils.layer_stream_runtime import RamSource

        shards, index, specs = _shards(tmp_path)
        # The stats are readable only once the CUDA runtime is up; a pinned
        # allocation alone does not bring it up for the stats getter.
        torch.cuda.init()
        before = int(torch.cuda.host_memory_stats().get("active_bytes.current", 0))
        source = RamSource(shards, index.n_layers, specs, pin=True)
        after = int(torch.cuda.host_memory_stats().get("active_bytes.current", 0))
        assert after - before == source.pinned_bytes
        assert source.pinned_bytes >= source.nbytes


class TestTheRuntimeReportsPageLockedBytes:
    def test_stats_carry_pinned_bytes_beside_the_store_bytes(self, tmp_path):
        from soup_cli.utils.layer_stream_runtime import RamSource, StreamRuntime

        shards, index, specs = _shards(tmp_path)
        source = RamSource(shards, index.n_layers, specs, pin=False)

        class _Pool:
            n = 2
            nbytes = 10
            loads = 0

        runtime = StreamRuntime(
            pool=_Pool(),
            source=source,
            prefetcher=None,
            n_layers=index.n_layers,
            pinned=False,
            device="cpu",
        )
        stats = runtime.stats()
        assert stats["store_bytes"] == source.nbytes
        assert stats["pinned_bytes"] == 0
