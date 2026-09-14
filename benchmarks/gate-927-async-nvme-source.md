<!--
Working measurement record, published verbatim.

Like the records beside it, this is the log kept while the work happened, not a
report assembled afterwards: the reading that was wrong at 13:15 and corrected
at 13:35, the confound that turned a depth result into an ordering result, the
outlier that did not reproduce, and a negative result on the warm fixture all
stay in, in the order they occurred.

Hardware: RTX 5070 Laptop GPU (Blackwell sm_120, 8151 MiB by nvidia-smi /
8.518 GB by torch, driver 616.92, PCIe 5.0 x8 negotiated), Intel i9-14900HX,
31.7 GB DDR5-5600, Windows 11 Pro 26100. Two Samsung MZAL81T0HFLB (PM9B1-class)
NVMe, 954 GB each: disk 0 is C:, disk 1 is D:. Every TIMED read below comes
from the NF4 shard cache on C: (disk 0); D: holds only the bf16 source
checkpoint, which is read during sharding and was a cache hit throughout.
Stack: Python 3.12.10 · torch 2.14.0+cu130 · transformers 5.17.0 · peft 0.20.0
· bitsandbytes 0.50.2 · safetensors 0.8.0.
Soup: branch feat/async-nvme-source at 9e5ce63a (v0.75.0 + Tasks 1-5 of #927 +
the two harness commits described in section 1).
Harness: benchmarks/harness/stream_probe.py, plus benchmarks/harness/
layer0_wait.py for section 8. Raw per-point JSON for every block is under
benchmarks/results/probe-rtx5070/.
-->

# Gate record — #927: the async NVMe source, measured cold and warm

**Status: MEASURED, 2026-09-14, with one negative result and one confound that
changed the headline.** The cold disk tier got **2.1x–3.1x faster** depending on
which position-matched pair you take, and the win comes with the GPU leaving
idle clocks for the first time. **Warm, with the whole store in the page cache,
the async source is 1.19x SLOWER than the one it replaces** — measured against a
same-day control, not against a published number from another session. What
looked at first like a read-ahead *depth* effect turned out to be the page cache
warming across blocks, and is published as the ordering result it is.

Unit convention: **decimal GB**, matching the other records.

---

## 0. The question, and what it is measured against

Layer streaming's disk tier read each layer synchronously, on the compute
thread, through a memory map. `probe-rtx5070-what-bounds-streaming.md` measured
what that costs: §17 put a cold 70B-shaped NF4 step at **124.5 s** (4.1 tok/s,
0.57 GB/s) on an NVMe that reads 3.5+ GB/s, §17a found it unstable by an order
of magnitude between blocks, and §18a put the **read-free floor at ~14.25 s** at
seq 256. That record's verdict named the fix: "an asynchronous source with
sequential unbuffered reads and pinned staging, off the compute thread".

Tasks 1–5 of #927 built the first half of that — a background reader with its
own header parsing (no mmap), pinned host staging, and a `training.stream_read_ahead`
depth. This record asks whether the cold step actually got faster, and by how
much. The honest ceiling for this project alone is the ~14 s floor, not the RAM
tier's numbers.

**Two comparisons are available and they are not the same thing.**

* The **published** before: §17/§17a/§8, measured 2026-09-12, in a different
  session, on a card whose boost clock varies ~13% between sessions.
* A **same-day control**: the shipped synchronous `DiskSource`, re-measured in
  this session on this fixture. Section 1(d) says how that was made without
  touching `src/`.

Both are reported. Where they disagree, the same-day control wins, and §2 shows
exactly why that ruling mattered.

---

## 1. The harness had to change first

`stream_probe.py` is the instrument, and as it stood it could not have measured
this source honestly. Four changes, all in the harness, none in `src/`.

**(a) `--read-ahead N`,** default 2, passed to `build_streamed_model` beside
`buffers`. The source allocates `min(N, members)` staging slots per distinct
layer shape, so the run's own `store` line is independent evidence of the depth
actually allocated rather than of the flag that was typed. On the 70B fixture
that arithmetic checks out exactly: 2 × 441.43 MB decoder + 2 × 536.87 MB vocab
= **1.9566 GB**, which is the reported store at depth 2.

**(b) `pin=` no longer forces pageable staging on the disk tier.** The shipped
line was `pin=(args.tier == "ram" and not args.no_pin)`, correct while the disk
tier allocated a fresh tensor per call and had nothing to page-lock. The async
source reads ahead into reusable host staging, so that expression would have
measured the **pageable fallback** rather than the shipped path. `--no-pin` now
selects the pageable arm on either tier.

**(c) The `Instruments` replicas now call `_release_source`,** which the shipped
`LayerBufferPool.load_async` and `LargeLayerBufferPool.load_async` do and the
replicas did not. Under pinned staging the source refuses to lend a second layer
while the first is still on loan, so the replica as it stood raised at the second
layer — that refusal working as designed, not a probe bug. The four release
lines were added to both drift-guard needle sets at the same time. Checked both
ways: the needles are present in the shipped bodies, and a deliberately drifted
needle (`self.events[slot + 1]`) makes the guard raise.

**(d) `--control-sync-source`,** which is how the same-day control was made
without changing `src/`. The runtime no longer constructs `DiskSource` for
`tier='disk'`, so the switch replaces the one name `_build_source` imports
lazily with a subclass of the shipped `DiskSource` that accepts and ignores the
two keyword arguments the async source added. The buffer pool, the prefetcher
and the layer wrapper stay the shipped ones — exactly as before Task 5 — and
`_release_source` is duck-typed, so it is a no-op against a source with no
`release`. The control therefore isolates the **read path**, not a different
scheduler. Every run now prints and records the source class it actually built
(`source_class`, `control_sync_source`, `read_ahead` in the JSON) and refuses
when that disagrees with the flag, so a control block cannot be mistaken for a
measurement; `--tier ram` is refused up front.

### Verification before any cold block

Warm 7B, `--steps 2 --warmup 1`. These three are FUNCTIONAL verification, not
throughput evidence: another session's full pytest suite was running throughout
(baseline commit charge 28.46 GB of 39.44 GB, free physical 15.04 GB).

| arm | the run's own `store` line | step |
|---|---|---|
| disk, pinned (default) | `0.762 GB pinned on tier disk (disk 4.138 GB)` | 3.484 s |
| disk, `--no-pin` | `0.762 GB pageable on tier disk (disk 4.138 GB)` | 3.627 s |
| ram (control) | `4.138 GB pinned on tier ram (disk 0.000 GB)` | 0.819 s |

The first row could not have been produced before change (b): the disk tier
reports **pinned** staging.

---

## 2. The cold headline

Synthetic Llama-70B **shape** (random weights — timing only, never a quality
claim), NF4, 36.39 GB on disk against 20–25 GB of free physical RAM, so no page
cache can hold it. Batch 1 × seq 512, 6 timed steps after 2 warm-up,
**uninstrumented** (§4 says why that qualifier is load-bearing). Sharding was a
cache hit in every block (`shard 0.0 s`), so every block read the same store.

| block | order | source | step | tok/s | source rate | GPU clock |
|---|---|---|---|---|---|---|
| control, sync `DiskSource` | 2nd | pageable, no reader | **100.35 s** (89.6–105.7) | 5.10 | 0.701 GB/s | 180 → 180 MHz |
| async, `read_ahead 2` | 1st | 1.957 GB pinned staging | **48.09 s** (43.3–52.5) | 10.65 | 1.464 GB/s | 1792 → 2355 MHz |
| control, sync `DiskSource` | 6th | pageable, no reader | **92.25 s** | 5.55 | 0.763 GB/s | — |
| async, `read_ahead 2` (repeat) | 5th | 1.957 GB pinned staging | **30.28 s** (29.1–34.0) | 16.91 | 2.324 GB/s | 180 → 705 MHz |

Bytes moved is **70.38 GB per step** in every row (157 decoder loads + 2 large),
and peak VRAM is **4.378 GB allocated / 4.80 reserved** in every row — streaming
still bounds the weights exactly as before; only the read path changed.

**Finding 2 — the cold step is 2.09x faster at the early position and 3.05x
faster at the late one.** Both pairs are same-session, same fixture, same shape,
adjacent in run order. Against §17's published 124.5 s — the other first-in-
session block — it is 2.59x. The spread between those ratios is not noise, it is
§3.

**Finding 2a — the async source converts a warming page cache into throughput
and the synchronous one largely cannot.** Across the same warming, early to
late in this session, the control moved 100.35 → 92.25 s (**1.09x**) while the
async source moved 48.09 → 30.28 s (**1.59x**). That is consistent with the
shipped source being bound by per-page synchronous faults *on the compute
thread*, where even a cache hit costs a fault and the GPU still waits.

**Finding 2b — the GPU stops idling.** The control's SM clock is 180 → 180 MHz:
it never leaves idle across a 10-minute block, which is §17's signature
("210 MHz — the GPU is idling between layers"). The async blocks reach
1417–2355 MHz.

**Against the floor.** §18a put the read-free step at **14.25 s at seq 256**;
the numbers above are at **seq 512**, so this is not a like-for-like ratio and
is quoted as an order of magnitude, not a factor: the best cold step here is
30.28 s, so the read path is still roughly half the step. **Still read-bound,
much less so.**

---

## 3. The depth series is an ORDERING result, not a depth result

This is the part that changed the headline, and it is left in the order it
happened.

| order | depth | step (uninstrumented) | source rate | free phys at its baseline |
|---|---|---|---|---|
| 1st | `read_ahead 2` | 48.09 s | 1.464 GB/s | 19.95 GB |
| 2nd | (sync control) | 100.35 s | 0.701 GB/s | 19.50 GB |
| 3rd | `read_ahead 1` | 39.23 s | 1.794 GB/s | 24.99 GB |
| 4th | `read_ahead 4` | 33.59 s | 2.095 GB/s | ~25 GB |
| 5th | `read_ahead 2` **repeat** | **30.28 s** | 2.324 GB/s | ~24 GB |

Read the first four rows alone and depth 4 is the winner and depth 2 the worst.
Read the fifth and that collapses: **the same configuration is 48.09 s run first
and 30.28 s run last, 1.59x apart with nothing changed but position** — larger
than the entire spread across depths (33.59–48.09 s).

**Finding 3 — depths 1, 2 and 4 are not distinguishable on this evidence, and
what looked like a depth effect was the page cache warming across blocks.** The
brief anticipated "if depth changes nothing, say so"; the honest version is
stronger, because the confound would have produced a confident and wrong
recommendation. Free physical RAM grew across the sequence (Windows enlarged the
pagefile during the first control, §6), and with a 36.39 GB store against 20–25
GB of free RAM the cached fraction grows with it.

`read_ahead 8` was **not** run. The brief asks for it only if 4 still moves the
number, and after the repeat there is no evidence that any depth moves it.

Staging cost, which *is* a clean function of depth and is worth stating because
it is what a user pays: **1.515 GB at depth 1, 1.957 GB at depth 2, 2.839 GB at
depth 4** (441.43 MB per decoder slot; the two vocabulary groups hold one member
each and so take one slot apiece however deep the decoder runs).

---

## 4. The instrumentation is not always free here, and one outlier did not reproduce

`--step` runs each block twice, once plain and once with the CUDA-event
instrumentation on. On the RAM tier those agree to 0.06% (§3 of the probe
record). Here:

| block | plain | instrumented | ratio |
|---|---|---|---|
| sync control (cold) | 100.35 s | 104.95 s | 1.046x |
| async `read_ahead 1` | 39.23 s | 38.63 s | 0.98x |
| async `read_ahead 2` | 48.09 s | **545.66 s** | **11.35x** |
| async `read_ahead 4` | 33.59 s | 33.38 s | 0.99x |
| sync control (cold, repeat) | 92.25 s | 93.48 s | 1.013x |
| async `read_ahead 2` **repeat** | 30.28 s | 29.71 s | 0.98x |

**Correction, 2026-09-14 13:15 → 13:35.** On the strength of the first three
rows I wrote that the artifact was "specific to the async source" and localised
it to the read-ahead/release path — `_plan_queue`'s own docstring says the plan
is empty at depth 1, so depth 1 having no lookahead looked like the discriminator.
**`read_ahead 4` refutes that**: it has lookahead and release, and its
instrumentation is free. The localisation was wrong and is left standing with
this correction beside it.

**Finding 4 — the 545.66 s block is an outlier that did not reproduce.** The
same configuration instrumented, run last, is 29.71 s. §17a recorded the same
shape of thing on the shipped source (124 s plain against a 746 s instrumented
mean, with one step at 3129 s) and attributed it to the page cache. A targeted
pytest run belonging to another session appears in that block's AFTER stamp and
not in its BEFORE stamp, so contention is a candidate and is not established.
It is published as measured. Every headline in this record is an
**uninstrumented** block on both sides, which is like-for-like and unaffected
either way.

---

## 5. Warm, the async source is SLOWER — a negative result

Mistral-7B NF4, 4.138 GB store entirely in the page cache, batch 1 × seq 512,
8 timed after 3 warm-up, uninstrumented. Run in the order listed; the store is
4.1 GB against ~24 GB of free RAM, so all three were fully cached.

| arm | step | tok/s | source rate |
|---|---|---|---|
| async disk, `read_ahead 2`, pinned | **1.84 s** | 277.8 | 4.016 GB/s |
| control, shipped sync `DiskSource` | **1.54 s** | 333.6 | 4.822 GB/s |
| RAM tier, same session | **0.82 s** | 627.8 | 9.075 GB/s |

**Finding 5 — with the whole store in the page cache the async source costs
1.19x against the source it replaces.** Against §8's published warm before
(2.46 s, 177–208 tok/s) the same number would read as a **1.34x improvement**,
and that framing would be wrong: today's box is simply faster than that session.
**This is the single clearest argument for the same-day control**, and the
reason §8 is not used as the warm baseline here.

The reading is unsurprising once stated: with the store cached, a synchronous
`mmap` read is close to a `memcpy`, so there is nothing for a background reader
to hide, and the handoff, the staging copy and the release/drain synchronisation
are pure overhead. The async source exists for a store that does **not** fit
RAM. When it does fit, the RAM tier is the right answer anyway — 2.2x faster
than either disk arm here.

Ablation on the warm async source (2 interleaved rounds, uninstrumented):

| arm | round 0 | round 1 |
|---|---|---|
| A baseline | 1.71 s | 1.75 s |
| B no source read, no copy | **0.74 s** | 0.74 s |
| C no NF4 dequantisation | 1.79 s | 1.81 s |
| D neither | 0.60 s | 0.59 s |

Removing the read path buys 57% (§10 measured 65% on the shipped source in its
own session), and arm B at 0.74 s sits just under the RAM tier's 0.82 s, as it
should — B removes the device copy too.

---

## 6. Two incidental observations

**The commit limit moved, and only under the control.** Block 1's AFTER stamp
reads `commit charge 24.87 GB of 39.44 GB`; the first control's reads
`21.48 GB of 51.36 GB`. Windows grew the pagefile by ~12 GB **during the
control** and not during any async block. `DiskSource` memory-maps all 80 layer
shards and holds them for the run (36.39 GB), and a private mapping charges
commit for the file — #926 measured 48.99 GB of mappings raising the charge
46.17 GB. The async source never maps: it parses headers and `readinto`s a
pre-allocated buffer. That is the "no mmap" half of the spec showing up as a
side effect rather than as a designed measurement.

**The 30 s no-progress guard never fired.** No log from any block contains
"no progress", including the 545.66 s step. Reading the code, that is expected
rather than lucky: the raise in `get` is guarded by `self._in_flight != idx and
idx not in self._queue`, so a read that is merely slow keeps looping and only a
reader that has stopped making progress trips it. **The guess was not tested by
these runs** — nothing here establishes whether 30 s is the right number for a
reader that genuinely stalls.

---

## 7. `read_ahead` staging, for the operator

| depth | staging on a 70B (441.43 MB/layer) | staging on a 7B (246.7 MB/layer) |
|---|---|---|
| 1 | 1.515 GB | 0.516 GB |
| 2 (default) | 1.957 GB | 0.762 GB |
| 4 | 2.839 GB | — |

Each level costs one more decoder layer of **pinned** host memory. Since no
depth was distinguishable on throughput here (§3), the default of 2 is not
challenged by this record, and 1 is the cheaper choice if host RAM is tight.

---

## 8. Layer 0 at the step head is not material

`_plan_queue` walks one index out of a single-member group, so a
vocabulary-sized read is planned ahead of decoder layer 0 at the moment the
compute thread blocks on layer 0. `benchmarks/harness/layer0_wait.py` drives the
probe's own instruments and labels each load bracket with the layer it belongs
to. Cold 70B, depth 2, three steps, no warm-up, instrumented from step 0 — so
its **absolute** step times (65.0, 55.0, 52.9 s) are not throughput evidence;
the ratio is what it measures.

| step | layer 0, forward / backward | other layers, mean / median / max | vocabulary loads |
|---|---|---|---|
| 0 | **28** / 455 ms | 363 / 401 / 592 ms | 874, 513 ms |
| 1 | **25** / 389 ms | 327 / 338 / 504 ms | 41, 43 ms |
| 2 | **25** / 344 ms | 313 / 322 / 549 ms | 32, 43 ms |

**Finding 8 — layer 0's forward load is the FASTEST load of the step**, 25–28 ms
against a 313–363 ms mean, because the reader has the whole build and embedding
phase to stage it before the compute thread asks for it. Its backward visit is
ordinary. The vocabulary read costs 874 ms exactly once, on the first touch of
the first step, then 32–43 ms. The addendum offered a follow-up (refuse to plan
out of a single-member group); **on this evidence it is not needed**, and that
is scoped to one fixture, one depth, one shape.

The same table carries the step's own accounting: the per-load brackets are
**90–93% of the step**. This step is the read path.

---

## 9. Contention, block by block

The box is shared with a contributor-PR review session that runs test suites
and cannot be stopped. Every block stamps the commit charge, the free physical
memory and the other Python processes immediately before and after itself.

| block | baseline commit charge | free phys | other work |
|---|---|---|---|
| §1 verification (3 arms) | 28.46 GB of 39.44 GB | 15.04 GB | **a full pytest suite throughout** |
| async `read_ahead 2` | 21.66 GB of 39.44 GB | 19.95 GB | none at start; **a targeted pytest in the AFTER stamp** |
| control (early) | 21.82 GB of 39.44 GB | 19.50 GB | none at start; 1 pytest by 12:55, 3 by 12:59 |
| async `read_ahead 1` | 21.54 GB of 51.36 GB | 24.99 GB | none |
| async `read_ahead 4` | 21.38 GB of 51.36 GB | 24.88 GB | none |
| async `read_ahead 2` repeat | ~21 GB of 51.36 GB | ~24 GB | none |
| control (late) | ~21 GB of 51.36 GB | ~24 GB | none |
| warm 7B, three arms | ~21 GB of 51.36 GB | ~24 GB | none |
| layer-0 probe | 25.46 GB of 51.33 GB | 20.40 GB | none |

The early control ran with pytest present and the early async block did not,
which biases that pair **in favour of the reported speedup**. The late pair
(§2, rows 3–4) is clean on both sides, which is why it is the one quoted as the
position-matched result.

---

## 10. Verdict

The cold disk tier is no longer bound by synchronous page faults on the compute
thread. Position-matched and same-session, a cold 70B-shaped NF4 step went from
**92.25 s to 30.28 s (3.05x)** at the late position and from **100.35 s to
48.09 s (2.09x)** at the early one, with the source rate rising from 0.70–0.76
GB/s to 1.46–2.32 GB/s and the GPU leaving idle clocks for the first time on
this tier. Against §17's published 124.5 s it is 2.59x. **What bounds it now is
still the read**: the per-load brackets are 90–93% of the step (§8), and §18a's
read-free floor — at a shorter sequence, so an order of magnitude rather than a
factor — is ~14 s against a best measured 30.28 s. The remaining levers are the
ones §19 of the probe record already named and this project did not touch:
unbuffered sequential reads, both NVMe drives, layer-major micro-batching, and
#842.

Two things temper it. **Warm, this is a regression** (§5): with the store in the
page cache the async source costs 1.19x against the one it replaces, and the
right answer there is the RAM tier. And **no read-ahead depth was
distinguishable** (§3) once the ordering confound was controlled, so the depth
knob is, on this evidence, a staging-cost knob rather than a throughput one.

---

## 11. What was NOT measured

- **A real 70B.** The cold fixture is a synthetic Llama-70B *shape* with random
  weights. Nothing here is a correctness claim; the bit-exactness gates are
  Task 3's and were not re-run.
- **Correctness of any kind.** This record is timing only.
- **The NVMe itself.** No block-level read test; 3.5+ GB/s sequential is the
  drive's published figure, not measured here.
- **A second drive, striping, unbuffered/O_DIRECT reads, or #842.** All four are
  named in §19 of the probe record as the remaining levers and none is in #927.
- **`read_ahead 8`,** and any depth on the warm fixture (§3 says why 8 was
  dropped).
- **Any shape other than batch 1 × seq 512,** and any sequence sweep. The
  read-free floor this is compared against was measured at seq 256/128.
- **Whether 30 s is the right no-progress timeout.** It never fired (§6); a
  reader that genuinely stalls was not constructed.
- **Any architecture beyond llama (the 70B shape) and mistral,** any OS other
  than Windows 11, any card other than this one, and any tier interaction with
  `stream_source: auto`'s own RAM-vs-disk decision.
- **A pageable-staging cold arm.** `--no-pin` was verified to work on the disk
  tier (§1) but no cold block was run with it, so the value of pinning *on this
  tier* is unquantified.
