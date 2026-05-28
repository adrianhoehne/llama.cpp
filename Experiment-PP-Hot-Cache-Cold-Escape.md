# Experiment PP Hot Cache Cold Escape

Status: 2026-05-28

## Executive Summary

This experiment did not find a useful PP switch-over point for the current cold-escape path. For long prompt processing, the Hot/Worklist path is faster than the Cold-Escape/Default-Cold comparison path, even when only a very small number of slots are in the hot cache.

The important correction to the early assumption is this: `LLAMA_MOE_HOT_CACHE_PP_MIN_HOT_SLOT_RATIO=0.30` technically means "escape when fewer than 30% of TopK slots are hot", but the benchmarks do not show that the cold path is faster below 30%. In fact, on PP3070 Hot/Worklist is equal or faster down to below 10% hot slots.

Recommendation for the current state:

- Do not enable PP Cold-Escape aggressively.
- Do not use `0.30` as a default or recommendation.
- If enabled at all, use it only experimentally and very conservatively below roughly `0.08`; even there, the PP3070 gain is not proven.
- The real benefit would only be expected if Cold-Escape can jump directly into a less redundant Default-Cold subgraph and no longer carries the same overhead it does now.

## Branch Deletion Summary

This branch was an experiment for Qwen3.6/Qwen35 MoE Hot Cache during prompt processing. The changes were not intended as a final product path. They were meant to test whether low hot-cache coverage can be detected early enough to switch into a cold path.

High-level changes:

- Added a dynamic decision after TopK/Routing in the Hot-Cache graph: enough hot slots -> Hot/Worklist path, too few hot slots -> Cold-Escape branch.
- Added hot-slot-ratio environment variables for that decision, especially `LLAMA_MOE_HOT_CACHE_PP_MIN_HOT_SLOT_RATIO`.
- Built a separate Qwen Cold-Escape MoE branch that reuses existing `selected_experts` and `weights`.
- Experimentally extended the ggml scheduler so nested Hot/Cold parallel regions and mixed join backends can work.
- Extended Worklist, Perf-JSON, and tests to measure hot-slot ratio, branch activity, and Cold-Escape behavior.

Important takeaway:

- The current implementation does not prove that the cold path is faster at low hot coverage.
- On PP3070, Hot/Worklist was faster or equal even at very low ratios.
- `0.30` is the wrong PP switch-over threshold; it would send many fast Hot/Worklist cases into a slower cold path.
- The branch can be deleted if these findings and benchmark results are preserved.

If this is restarted later, this concrete Cold-Escape implementation should not be continued as-is. A better approach would be a fresh design that really enters a middle Default-Cold subgraph and avoids Worklist/Merge overhead, instead of building a separate Cold-Escape branch next to the hot path.

## Implementation State

The goal was a rebase-friendly Qwen-first approach inside the `moe-hot-cache` area, with the smallest practical hook in the model graph.

Implemented:

- Dynamic hot-slot decision after TopK/Routing:
  - `selected_experts` and `weights` are computed once.
  - A CPU map op then decides from the real hot-slot ratio per layer/chunk whether Hot/Worklist or Cold-Escape should be active.
- New policy environment variables:
  - `LLAMA_MOE_HOT_CACHE_MIN_HOT_SLOT_RATIO`
  - `LLAMA_MOE_HOT_CACHE_PP_MIN_HOT_SLOT_RATIO`
  - `LLAMA_MOE_HOT_CACHE_TG_MIN_HOT_SLOT_RATIO`
  - `LLAMA_MOE_HOT_CACHE_DECODE_MIN_HOT_SLOT_RATIO`
  - The default remains `0.0`, meaning disabled.
- Qwen35/Qwen36 Cold-Escape graph inside the Hot-Cache code:
  - builds the MoE FFN using existing `selected_experts` and `weights`;
  - pins the expert matmul part of the Cold-Escape to CPU;
  - joins the result back into the main graph.
- Scheduler support for nested Hot/Cold regions:
  - `GGML_BACKEND_SCHED_MOE_HOT_CACHE_PARALLEL_FLAG_ALLOW_JOIN_BRIDGE`
  - `GGML_BACKEND_SCHED_MOE_HOT_CACHE_PARALLEL_FLAG_ALLOW_MIXED_LANES`
  - needed because the outer Cold-Escape decision region can contain an inner Hot/Cold parallel region.
- Worklist/Perf extensions:
  - hot-slot counters in the Decision/Worklist path;
  - Perf-JSON remains usable for hot-slot ratio, launches, merge/worklist time, and layer activity.
- The older static expert-ratio idea was not kept as an active default rule.
- Tests were adjusted/extended:
  - `tests/test-moe-hot-cache-pp.cpp`
  - `tests/test-moe-hot-cache-worklist.cpp`

Important: the current implementation is not yet the ideal "entry point in the middle of the original Default-Cold path". It avoids recomputing TopK/weights, but still builds a separate Cold-Escape branch. The benchmarks show exactly why this does not help PP.

## Changed Files

Diff stat at the time of this summary:

```text
docs/moe-hot-cache/moe-hot-cache-pp-journey.html   |  32 +--
ggml/include/ggml-backend-moe-hot-cache.h          |   1 +
ggml/src/ggml-backend-moe-hot-cache.inc            | 253 ++++++++++++++++---
src/models/qwen35moe.cpp                           |   7 +-
src/moe-hot-cache/llama-moe-hot-cache-graph.cpp    | 268 +++++++++++++++++----
src/moe-hot-cache/llama-moe-hot-cache-pp.cpp       |  56 ++---
src/moe-hot-cache/llama-moe-hot-cache-pp.h         |   9 +-
src/moe-hot-cache/llama-moe-hot-cache-worklist.cpp |  68 +++++-
src/moe-hot-cache/llama-moe-hot-cache.h            |  27 +++
tests/test-moe-hot-cache-pp.cpp                    |  56 ++---
tests/test-moe-hot-cache-worklist.cpp              |  55 +++++
11 files changed, 653 insertions(+), 179 deletions(-)
```

## Verification

Build/test before the benchmarks:

```bash
cmake --build build --target test-moe-hot-cache-pp test-moe-hot-cache-worklist llama-bench -j2
ctest --test-dir build -R '^test-moe-hot-cache-' --output-on-failure
git diff --check
```

Result:

- Build succeeded.
- `ctest`: 16/16 passed.
- `git diff --check`: no whitespace errors.

## Benchmark Setup

Model:

```text
/home/adrian/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf
```

Prompt:

```text
pp-bench-conversation-code.txt
```

Standard command shape:

```bash
./build/bin/llama-bench \
  -m "$MODEL" \
  --prompt-file pp-bench-conversation-code.txt \
  -p 3070 \
  -n 0 \
  -r 1 \
  --device CUDA0 \
  --progress
```

Hot-cache flags:

```bash
--cpu-moe \
--moe-hot-cache qwen36 \
--moe-hot-cache-weighting flat \
--moe-hot-cache-layer-curve 0.7 \
--moe-hot-cache-pp-reduce-merge on
```

Full-cache auto budget for the early runs:

```bash
--moe-hot-cache-max-mib -1 \
--moe-hot-cache-auto-reserve-mib 3000
```

Forced Cold-Escape:

```bash
LLAMA_MOE_HOT_CACHE_PP_MIN_HOT_SLOT_RATIO=1.0
```

## Benchmark 1: PP Reduce Merge Baseline Comparison

Result dir:

```text
bench-pp-hot-cache-results/run-20260527-204231
```

| Variant | Test | Tokens/s |
|---|---:|---:|
| Baseline without Hot Cache | pp3070 | `75.65 +/- 0.00` |
| Hot Cache, PP reduce merge off | pp3070 | `86.83 +/- 0.28` |
| Hot Cache, PP reduce merge on | pp3070 | `106.64 +/- 0.18` |

Result: PP reduce merge is clearly required for this prompt and brings the Hot-Cache path into the `106 t/s` range.

## Benchmark 2: Early Standard-Cold Sweeps

Result dirs:

```text
bench-pp-hot-cache-results/run-20260527-210106-pp-standard-cold-sweep
bench-pp-hot-cache-results/run-20260527-211952-pp-standard-cold-fine-sweep
```

| Variant | Test | Tokens/s |
|---|---:|---:|
| Standard cold baseline | pp3070 | `75.68 +/- 0.00` |
| Hot full, threshold 0 | pp3070 | `107.40 +/- 0.56` |
| Hot full, min expert 0.20 | pp3070 | `107.62 +/- 0.42` |
| Hot full, min expert 0.25 | pp3070 | `60.75 +/- 0.00` |
| Hot full, min expert 0.30 | pp3070 | `60.76 +/- 0.01` |
| Hot full, min expert 0.35 | pp3070 | `60.76 +/- 0.00` |
| Hot full, min expert 0.21 | pp3070 | `108.04 +/- 1.08` |
| Hot full, min expert 0.22 | pp3070 | `107.62 +/- 0.54` |
| Hot full, min expert 0.23 | pp3070 | `107.75 +/- 0.42` |
| Hot full, min expert 0.24 | pp3070 | `60.76 +/- 0.00` |

Result: the old implementation had a hard edge around `0.24`. That was a different ratio/implementation and is not directly transferable to the new hot-slot decision path.

## Benchmark 3: 0.23 Comparison with the Old Ratio Idea

Result dir:

```text
bench-pp-hot-cache-results/run-20260528-091401-pp-min-expert-023-compare
```

| Variant | Test | Tokens/s |
|---|---:|---:|
| Baseline `-ncmoe31` | pp3070 | `75.69 +/- 0.01` |
| Hot Cache, min expert 0.23 | pp3070 | `107.29 +/- 1.03` |
| Forced standard cold escape | pp3070 | `60.75 +/- 0.00` |

Result: `0.23` was conservative and fast in the old logic, but the Forced-Cold path was much slower than `-ncmoe31`.

## Benchmark 4: New Slot-Ratio Cold-Escape, First State

Result dir:

```text
bench-pp-hot-cache-results/run-20260528-103201-pp-slot-escape-new
```

| Variant | Test | Tokens/s |
|---|---:|---:|
| Baseline `-ncmoe31` | pp3070 | `75.70 +/- 0.00` |
| Hot Cache, no slot escape | pp3070 | `107.12 +/- 0.24` |
| Hot Cache, slot ratio 0.23 | pp3070 | `94.93 +/- 0.47` |
| Hot Cache, forced cold escape | pp3070 | `58.61 +/- 0.02` |

Micro PP128 from the same development state:

| Variant | Test | Tokens/s |
|---|---:|---:|
| Baseline `-ncmoe31` | pp128 | `31.37 +/- 0.01` |
| Hot Cache, no slot escape | pp128 | `67.93 +/- 4.32` |
| Hot Cache, slot ratio 0.23 | pp128 | `63.05 +/- 3.64` |
| Hot Cache, forced cold escape | pp128 | `56.12 +/- 2.87` |

Result: the new slot escape worked technically, but the Forced-Cold path was not attractive.

## Benchmark 5: Fix1 / Scheduler and Escape Fixes

Result dir:

```text
bench-pp-hot-cache-results/run-20260528-105641-pp-slot-escape-fix1
```

| Variant | Test | Tokens/s |
|---|---:|---:|
| Hot Cache, no slot escape | pp3070 | `108.50 +/- 0.42` |
| Hot Cache, slot ratio 0.23 | pp3070 | `97.50 +/- 0.39` |
| Hot Cache, slot ratio 0.23 perf | pp3070 | `97.96 +/- 0.00` |
| Hot Cache, forced cold escape | pp3070 | `59.10 +/- 0.32` |
| Hot Cache, forced cold escape perf | pp3070 | `58.54 +/- 0.00` |

Result: the path became more stable, but `0.23` still cost performance compared to no-escape.

## Benchmark 6: Final Check after Nested Region and Join-Backend Fix

Result dir:

```text
bench-pp-hot-cache-results/run-20260528-113443-pp-slot-escape-finalcheck
```

PP3070:

| Variant | Test | Tokens/s |
|---|---:|---:|
| `--cpu-moe` baseline | pp3070 | `60.76 +/- 0.00` |
| `-ncmoe31` baseline | pp3070 | `75.70 +/- 0.01` |
| Hot Cache, no slot escape | pp3070 | `107.89 +/- 0.23` |
| Hot Cache, slot ratio 0.23 | pp3070 | `102.81 +/- 0.31` |
| Hot Cache, forced cold escape | pp3070 | `59.15 +/- 0.02` |

Micro PP128:

| Variant | Test | Tokens/s |
|---|---:|---:|
| `--cpu-moe` baseline | pp128 | `25.59 +/- 0.00` |
| `-ncmoe31` baseline | pp128 | `31.39 +/- 0.01` |
| Hot Cache, no slot escape | pp128 | `69.72 +/- 3.16` |
| Hot Cache, slot ratio 0.23 | pp128 | `61.93 +/- 2.97` |
| Hot Cache, forced cold escape | pp128 | `56.14 +/- 1.98` |

Result: after the fixes, no-escape is fast. `0.23` is safe but slower than no-escape. Forced-Cold is roughly at `--cpu-moe` level for PP3070 and therefore much slower than Hot/Worklist.

## Benchmark 7: Full-Cache Ratio Calibration

Result dir:

```text
bench-pp-hot-cache-results/run-20260528-122741-ratio-calibration-pp3070
```

No-escape Full-Cache:

| Variant | Test | Tokens/s |
|---|---:|---:|
| Hot Cache, no escape perf | pp3070 | `107.85 +/- 0.00` |

Hot-slot distribution in no-escape Full-Cache:

- 40 layers
- Average hot-slot ratio: `56.60%`
- Lowest per-layer average: `33.33%`
- Highest per-layer average: `68.41%`
- Lowest layers: `2=33.33%`, `0=37.92%`, `1=40.46%`, `3=46.72%`, `11=47.09%`

Ratio sweep:

| PP min hot slot ratio | Tokens/s | Observation |
|---:|---:|---|
| 0.23 | `102.82` | safe, no relevant Cold-Escape activity |
| 0.30 | `102.84` | still safe |
| 0.305 | `103.72` | still safe |
| 0.310 | `67.22` | tips over, many escapes |
| 0.315 | `67.12` | too aggressive |
| 0.320 | `67.11` | too aggressive |
| 0.325 | `66.82` | too aggressive |
| 0.33 | `67.20` | too aggressive |
| 0.35 | `66.70` | too aggressive |
| 0.40 | `60.65` | near Forced-Cold level |
| 0.45 | `60.74` | near Forced-Cold level |
| 0.50 | `60.89` | Forced-Cold level |
| 0.55 | `60.75` | Forced-Cold level |
| 0.60 | `60.77` | Forced-Cold level |
| 0.65 | `60.83` | Forced-Cold level |
| 0.70 | `60.57` | Forced-Cold level |
| 1.00 | `59.07` | Forced-Cold |

Result: with Full-Cache, `0.30` is an upper safe bound because almost no real <30% cases occur. This does not prove that cold is faster below 30%.

## Benchmark 8: Budget Mapping PP512

Result dir:

```text
bench-pp-hot-cache-results/run-20260528-131324-budget-map-pp512
```

Goal: reduce the hot-cache budget to create real low hot-slot ratios.

| Budget | Hot-Slot Ratio | Hot/Worklist Tokens/s |
|---:|---:|---:|
| 64 MiB | `1.34%` | `56.43` |
| 128 MiB | `2.41%` | `56.43` |
| 256 MiB | `4.68%` | `58.40` |
| 512 MiB | `9.56%` | `60.95` |
| 768 MiB | `13.46%` | `63.20` |
| 1024 MiB | `16.50%` | `63.99` |
| 1536 MiB | `20.83%` | `68.12` |
| 2048 MiB | `24.32%` | `70.22` |
| 3072 MiB | `30.69%` | `74.65` |

Result: suitable A/B points for real low ratios were found.

## Benchmark 9: Budget A/B PP3070

Result dir:

```text
bench-pp-hot-cache-results/run-20260528-131733-budget-ab-pp3070
```

Baseline:

| Variant | Test | Tokens/s |
|---|---:|---:|
| `-ncmoe31` | pp3070 | `75.67 +/- 0.00` |

Budget A/B:

| Budget | Hot-Slot Ratio | Hot/Worklist | Forced-Cold | Hot minus Cold | Active Cold Layers |
|---:|---:|---:|---:|---:|---:|
| 64 MiB | `1.86%` | `59.09` | `116.45` | `-57.36` | 11 |
| 128 MiB | `3.10%` | `57.61` | `75.67` | `-18.06` | 23 |
| 256 MiB | `5.35%` | `57.33` | `52.21` | `+5.12` | 40 |
| 384 MiB | `8.89%` | `58.46` | `58.20` | `+0.26` | 40 |
| 512 MiB | `12.01%` | `61.29` | `59.09` | `+2.20` | 40 |
| 1024 MiB | `21.65%` | `67.15` | `58.93` | `+8.22` | 40 |
| 1536 MiB | `28.08%` | `71.99` | `59.04` | `+12.95` | 40 |
| 2048 MiB | `32.62%` | `74.91` | `58.76` | `+16.15` | 40 |
| 3072 MiB | `39.81%` | `83.29` | `58.51` | `+24.78` | 40 |

Note on 64/128 MiB: these points should not be treated as normal ratio calibration data, because only 11 and 23 layers had an active Hot-Cache path. For a clean hot-slot-ratio statement, the 40-layer points from 256 MiB onward are more relevant.

Result: Forced-Cold does not win on PP3070. From roughly 12% hot slots onward, Hot/Worklist is clearly faster. Around 9%, it is effectively a tie with a slight Hot advantage.

## Benchmark 10: Microbenchmark PP512

Result dir:

```text
bench-pp-hot-cache-results/run-20260528-135215-budget-ab-pp512
```

| Budget | Hot-Slot Ratio | Hot/Worklist | Forced-Cold | Hot minus Cold | Active Cold Layers |
|---:|---:|---:|---:|---:|---:|
| 256 MiB | `4.68%` | `58.55` | `60.07` | `-1.52` | 40 |
| 384 MiB | `7.19%` | `59.02` | `59.68` | `-0.66` | 40 |
| 512 MiB | `9.56%` | `59.68` | `58.34` | `+1.34` | 40 |

Result: on short PP512, Forced-Cold can be slightly faster at roughly 5-7%. Around 10%, Hot/Worklist wins again. This is not enough for a default rule for long PP.

## Benchmark 11: Focused PP3070 r=3 Check

Result dir:

```text
bench-pp-hot-cache-results/run-20260528-135601-budget-confirm-pp3070-r3
```

| Budget | Hot-Slot Ratio | Hot/Worklist | Forced-Cold |
|---:|---:|---:|---:|
| 384 MiB | `8.89%` | `59.13 +/- 0.68` | `58.22 +/- 0.72` |
| 512 MiB | `12.01%` | `60.00 +/- 0.89` | `58.78 +/- 0.17` |

Result: the r=3 run confirms the PP3070 finding. Forced-Cold does not win reliably even at the low edge.

## Additional Debug and Intermediate Benchmarks

These runs were development/validation runs during the scheduler and Cold-Escape fixes. They are not the final decision data, but they show which changes moved in which direction.

| Result Dir | Variant | Test | Tokens/s | Purpose |
|---|---|---:|---:|---|
| `run-20260528-111235-pp-slot-escape-nested` | slot ratio 0.23 perf | pp3070 | `103.58 +/- 0.00` | first nested-region check |
| `run-20260528-111524-pp-slot-escape-nested-defaultsched` | forced cold escape perf | pp3070 | `56.82 +/- 0.00` | default-scheduler comparison |
| `run-20260528-111524-pp-slot-escape-nested-defaultsched` | slot ratio 0.23 perf | pp3070 | `107.60 +/- 0.00` | checked whether nested region stays stable without Forced-Cold |
| `run-20260528-112020-pp-slot-escape-expertonly` | forced cold escape perf | pp3070 | `58.99 +/- 0.00` | Cold-Escape with only expert matmul pinned to CPU |
| `run-20260528-112020-pp-slot-escape-expertonly` | slot ratio 0.23 perf | pp3070 | `103.20 +/- 0.00` | expert-only escape with ratio 0.23 |
| `run-20260528-113103-pp-slot-escape-expertonly-joinbackend` | forced cold escape perf | pp3070 | `59.04 +/- 0.00` | join-backend fix |
| `run-20260528-113103-pp-slot-escape-expertonly-joinbackend` | slot ratio 0.23 perf | pp3070 | `103.17 +/- 0.00` | join-backend fix with ratio 0.23 |
| `run-20260528-112353-standard-cold-check` | `--cpu-moe` baseline | pp3070 | `60.76 +/- 0.00` | checked pure CPU-MoE cold path |
| `run-20260528-112353-standard-cold-check` | `-ncmoe31` baseline | pp3070 | `75.71 +/- 0.00` | confirmed VRAM/GPU-layer baseline |
| `run-20260528-112737-hot-cache-ncmoe-check` | hot cache + `-ncmoe31` | pp3070 | `47.86 +/- 0.00` | showed that this combination is not a useful Hot-Cache baseline |

Complete result dirs in the worktree:

```text
bench-pp-hot-cache-results/run-20260527-203916
bench-pp-hot-cache-results/run-20260527-204231
bench-pp-hot-cache-results/run-20260527-210106-pp-standard-cold-sweep
bench-pp-hot-cache-results/run-20260527-211952-pp-standard-cold-fine-sweep
bench-pp-hot-cache-results/run-20260528-091401-pp-min-expert-023-compare
bench-pp-hot-cache-results/run-20260528-103201-pp-slot-escape-new
bench-pp-hot-cache-results/run-20260528-105628-pp-slot-escape-fix1
bench-pp-hot-cache-results/run-20260528-105641-pp-slot-escape-fix1
bench-pp-hot-cache-results/run-20260528-111235-pp-slot-escape-nested
bench-pp-hot-cache-results/run-20260528-111524-pp-slot-escape-nested-defaultsched
bench-pp-hot-cache-results/run-20260528-112020-pp-slot-escape-expertonly
bench-pp-hot-cache-results/run-20260528-112353-standard-cold-check
bench-pp-hot-cache-results/run-20260528-112737-hot-cache-ncmoe-check
bench-pp-hot-cache-results/run-20260528-113103-pp-slot-escape-expertonly-joinbackend
bench-pp-hot-cache-results/run-20260528-113443-pp-slot-escape-finalcheck
bench-pp-hot-cache-results/run-20260528-122741-ratio-calibration-pp3070
bench-pp-hot-cache-results/run-20260528-131324-budget-map-pp512
bench-pp-hot-cache-results/run-20260528-131733-budget-ab-pp3070
bench-pp-hot-cache-results/run-20260528-135215-budget-ab-pp512
bench-pp-hot-cache-results/run-20260528-135601-budget-confirm-pp3070-r3
```

## Interpretation

The ratio from the old implementation is not transferable to the new implementation:

- Old ratio: more of an expert/layer-oriented bypass heuristic.
- New ratio: real TopK slot coverage per layer/chunk after routing.

`0.30` was "safe" in the Full-Cache curve only because almost no chunks in the Full-Cache case were actually below 30%. Once real low hot-slot ratios are created by small budgets, the result is:

- On PP3070, Hot/Worklist is faster than Forced-Cold even at `5.35%`.
- At `8.89%`, it is effectively a tie with a slight Hot advantage.
- At `12.01%` and above, Hot/Worklist is clearly faster.
- At `28.08%`, Cold-Escape would be clearly wrong.

Therefore the answer to the central question is:

```text
No, below 30% hot slots the current cold path is not faster.
```

## Consequences for Future Work

This experiment was still useful because it ruled out a wrong optimization direction:

- The current PP Cold-Escape is not suitable as a default optimization.
- The intended overhead saving from "no Worklist/no Merge" is not realized by the current implementation.
- The separate Cold-Escape branch is too expensive, or at least not better than Hot/Worklist.
- A real win requires the cold path to enter a Default-Cold subgraph earlier and more cleanly, after Routing/TopK/Weights have already been computed.

Next useful direction:

1. Keep the current PP Cold-Escape disabled by default.
2. Investigate TG/Decode separately, because its cost structure is different.
3. If PP is pursued further, do not tune the ratio. Rework the cold subgraph so it actually performs less work and has less Join/Merge overhead.

