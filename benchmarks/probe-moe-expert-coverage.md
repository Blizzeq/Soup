<!--
Working measurement record, published verbatim. The decision rule in §2 was
written and committed BEFORE any model was downloaded, which is the whole point
of it: a rule written after the numbers are in is not a rule.

Hardware: RTX 5070 Laptop GPU (Blackwell sm_120, 8151 MiB), Intel i9-14900HX,
31.7 GB DDR5-5600, two Samsung PM9B1 NVMe, Windows 11 Pro 26100. Stack: Python
3.12.10, torch 2.14.0+cu130, transformers 5.17.0, bitsandbytes 0.50.2.
Soup: branch probe/moe-expert-coverage cut from origin/main a08ae74f, worktree
C:\Users\user\projects\Soup-moe.
Harness: benchmarks/harness/moe_expert_coverage.py (committed with this file).
-->

# Probe — does a training step touch every expert? (MoE step 0)

**Status: the instrument is built and validated; NO MODEL HAS BEEN MEASURED
YET.** This file is committed in that state on purpose. `.claude/plan.md` asks
for a decision rule "written before the run", and the only way to make that
checkable is to put it in version control before the run happens. §5 onward is
empty and will be filled by the measurement.

---

## 0. The question, and why it decides a feature

Layer streaming today bounds VRAM by ONE decoder layer times `stream_buffers`.
On a 744B-class MoE one NF4 layer is ~5 GB, so two buffers do not fit an 8 GB
card and the model is refused at the pre-flight. Streaming an EXPERT at a time
instead would bound it by "the dense part plus a few experts", which on paper
admits any size.

Whether that is worth building turns on one number, and `.claude/plan.md` states
the trap plainly: if a step's token batch routes to nearly every expert anyway,
**the read volume does not fall** and only the memory bound improves. Our step
reads the stack twice (forward, then the backward recompute), so on a 744B store
that is ~600 GB per step; at the 4.5-5.1 GB/s the disk tier reaches after #974
that is around two minutes per step even if nothing else got worse.

So: **union coverage** — the fraction of a layer's experts that at least one
token in the step routes to — measured per layer, on a real model, on real text,
at the batch shapes Soup actually trains at.

## 1. Why a coverage number alone means nothing

With `E` experts, `k` picked per token and `T` tokens routed independently and
uniformly, the expected coverage is

```
1 - (1 - k/E)^T
```

For OLMoE's 64 experts / top-8 at T = 512 tokens that is `1 - (7/8)^512`, which
is `1 - 10^-29.7`. **Chance alone predicts coverage indistinguishable from
100%.** Even a single sequence of 512 tokens is far more than enough to touch
every expert if routing is anywhere near uniform.

A measured coverage of 0.95 is therefore not "most experts are used"; it is a
*large* departure from chance in the direction that helps. Every coverage figure
in §5 is printed beside this baseline, and the harness computes it.

This also sets the bar honestly: for the read to fall by half, routing must be
concentrated enough that half the experts see **no token at all** out of 512 —
roughly 4,096 assignments landing on at most 32 of 64 experts.

### The baseline across the whole family, computed before any model was loaded

Pure arithmetic from the formula above, so it needs no hardware and no download,
and it is what makes the bar concrete. The last two columns are the token budget
at which uniform routing would reach 50% and 99% coverage.

| model | E | k | baseline at T=512 | at T=2048 | T for 50% | T for 99% |
|---|---|---|---|---|---|---|
| `OLMoE-1B-7B` (measured here) | 64 | 8 | 1 - 10^-29.7 | 1 - 10^-119 | 5.2 | 34 |
| `granite-3.0-1b-a400m` (measured here) | 32 | 8 | 1 - 10^-64.0 | 1 - 10^-256 | 2.4 | 16 |
| `Qwen3-30B-A3B` (plan; not measured, §7) | 128 | 8 | 1 - 10^-14.4 | 1 - 10^-57 | 10.7 | 71 |
| `Mixtral-8x7B` | 8 | 2 | 1 - 10^-64.0 | 1 - 10^-256 | 2.4 | 16 |
| DeepSeek-V3 class | 256 | 8 | 1 - 10^-7.1 | 1 - 10^-28 | 21.8 | 145 |

**Read the last column.** Under uniform routing, even a 256-expert model needs
only **145 tokens** to touch 99% of its experts, and Soup's smallest streaming
step is 512. Expert-granularity streaming can only save READS if real routing is
concentrated enough that a 512-token step behaves like roughly five tokens'
worth of routing diversity. That is the size of the departure from chance the
feature needs, and stating it before the measurement is what keeps a coverage of
0.9 from being reported as encouraging.

A larger expert count does move the baseline in the helpful direction — the
whole column shifts right as E/k grows — which is precisely why §7 records that
the largest model measured here has 64 experts, and that the plan's own target
class (744B, 19,456 routed experts across the stack) is far outside it.

## 2. The decision rule, written before the run

Taken from `.claude/plan.md` and made specific. `C` is the mean union coverage
over layers at the shape Soup trains at (batch x seq = 512 and 2048); `S` is the
skew, reported as the share of assignments taken by the busiest 25% of experts.

| measured | verdict | what ships |
|---|---|---|
| **C <= 0.50** | both memory AND read win | schedule the feature: per-expert sharding, an expert buffer pool, the planner arithmetic in expert units. Size: the whole async-NVMe project again |
| **0.50 < C < 0.85** | ambiguous | do not schedule on this evidence. Re-measure at the shapes and models the decision would actually apply to, and state the read saving as `1 - C` with its spread, not as a headline |
| **C >= 0.85** | memory only | build it, if at all, as **"a capacity tier for MoE"** with an explicit promise of **no throughput gain** — the same honesty v0.72.3's 70B demonstration was given |
| **S >= 0.60** (busiest quartile takes 60%+ of traffic) | hot/cold split is real | a **hybrid RAM+disk tier** — hot experts resident, cold ones streamed — is the cheaper first deliverable and should precede full expert streaming, whatever C says |
| **hot-set Jaccard < 0.7 step to step** | the hot set is not stable | a persisted per-dataset heat file is NOT the input it looks like; the hybrid tier would have to re-learn continuously, which changes its cost |

**A third condition, which the plan does not state and which I am adding before
seeing any number.** Even at low coverage, a read saving only materialises if
the expert set can be known EARLY ENOUGH to prefetch. Today's prefetcher is a
perfect predictor because the walk is deterministic — forward 0..L-1, backward
L-1..0 — and `stream_read_ahead` exploits exactly that. Which *experts* a layer
needs is not known until its router has run, which is after the previous layer's
output exists. Expert-granularity therefore trades a perfect prefetch for a
conditional one, and a reader that must wait for the router is a reader that
stalls. Colibrì's answer is a lookahead that predicts the next layer's experts
from the current one (they report 71.6% one layer ahead). **That predictability
is NOT measured in step 0**, and until it is, a "C <= 0.50" result licenses
scheduling the feature's *design*, not a throughput claim.

## 3. What is measured, and on what

| model | experts / top-k | why it is here |
|---|---|---|
| `allenai/OLMoE-1B-7B-0924` | 64 / 8 | the plan names it; 6.9B total / 1.3B active, the smallest fully-trained MoE with a realistic expert count |
| `ibm-granite/granite-3.0-1b-a400m` | 32 / 8 | a second (E, k) point at a quarter the experts, cheap, and it fits the card in bf16 — so it is also the control that the NF4 arm is checked against |

Three corpora, because routing skew is a property of the text and a single
domain would overstate it:

- **prose** — `wikitext-2-raw-v1`, train split;
- **code** — this repository's own `src/soup_cli/`, which is real Python and
  needs no download;
- **math** — `gsm8k`, `main` config, train split, the question field.

Shapes: `1x512`, `4x512`, `1x2048` — the plan asks for batch x seq of 512 and
2048, and the two ways of reaching 2048 are measured separately because routing
is context-dependent and a 4x512 step is not a 1x2048 step.

## 4. The instrument, and what was done to trust it

`benchmarks/harness/moe_expert_coverage.py`, standalone (no Soup import, so it
runs against a stock transformers install).

Router discovery is by SHAPE rather than an architecture table: in transformers
5.17 every MoE decoder — olmoe, qwen3_moe, mixtral, granitemoe, deepseek\* —
exposes a `*TopKRouter` module returning `(router_logits, router_scores,
router_indices)` with integer indices of shape `(tokens, top_k)`. The harness
hooks anything of that shape, falls back to taking `topk` of a float
`(tokens, num_experts)` output itself, and **refuses by name** rather than
guessing if neither holds.

Tokens are **packed, not padded**: the corpus is tokenised into one stream and
sliced into exact `seq`-length chunks. A padded batch routes its PAD positions
too, and those assignments would land in the counts as if they were traffic.

Validated 2026-09-15 before any real model was downloaded
(`validate_moe_harness.py`, kept in the session scratchpad; it builds a 6.62M
-param OLMoE with random weights and drives the harness end to end):

- the statistics against hand-computed values — `uniform_coverage(8,2,4) =
  1-0.75^4`, `gini([1,1,1,1]) = 0`, `gini([0,0,0,4]) = 0.75`, `top_share`,
  `hot_set`, `jaccard`;
- index extraction: given `(logits, scores, indices)` it takes the integer
  tensor, given logits alone it takes the right top-k itself, and given neither
  shape it returns nothing rather than inventing an answer;
- **the arithmetic identity that is the real check**: the per-expert counts must
  sum to exactly `tokens x top_k x steps` at every layer. They do, at 3 layers x
  2 shapes. That is what proves no token was dropped or double-counted.

**One correction, kept because it is the point.** A first version of the
validation asserted that 512 tokens over 8 experts "must reach coverage 1.0, as
chance predicts". It measured 0.979, and the **assertion** was wrong, not the
harness: a randomly-initialised router is a fixed random map rather than a
uniform one, and the validation corpus is one sentence repeated, so the hidden
states barely vary and the routing concentrates (gini 0.43). The check now
asserts that shape — coverage below the baseline, with visible skew — which is
the behaviour the harness exists to detect.

### The NF4 caveat, and how it is bounded rather than assumed

OLMoE in bf16 is 13.8 GB and does not fit 8 GB of VRAM, so the primary arm loads
it as NF4 (`--load-4bit`). The router itself is **not** quantised: in
transformers 5.17 a top-k router holds its weight as a bare `nn.Parameter` used
through `F.linear`, not an `nn.Linear`, so `replace_with_bnb_linear` never sees
it. What quantisation can still move is the hidden state the router reads. That
is why a **bf16 control on CPU** — one corpus, one shape, few steps — is part of
the run rather than a footnote: the difference between the two arms bounds the
caveat with a number.

## 5. Results

*Not yet measured.*

## 6. Verdict

*Not yet measured. The rule in §2 decides it, and §2 is committed before §5
exists.*

## 7. What this will NOT measure

- **`Qwen3-30B-A3B`, which the plan names.** 60 GB in bf16 against 31.7 GB of
  RAM and 8 GB of VRAM; NF4 is ~15 GB, which fits neither the card nor a
  comfortable share of host RAM alongside a 60 GB download. It is skipped, and
  the two models in §3 are what this box can honestly carry. The consequence is
  stated rather than hidden: **the largest expert count measured here is 64**,
  and a 128- or 256-expert model at the same token budget would have a LOWER
  chance baseline and could behave differently.
- **One-layer-ahead expert predictability** — the precondition named in §2 for
  any throughput claim.
- **Any timing.** The forward runs with a hook on every router and, in the CPU
  control, at no throughput anybody should quote.
- **Backward routing.** Only the forward's router decisions are observed; the
  recompute re-runs the same routers on the same hidden states, so the expert
  SET is the same, but that is an argument rather than a measurement.
- **Training-time drift.** All routing here is from a trained checkpoint at rest.
  Whether a fine-tune moves the hot set is a different question.
