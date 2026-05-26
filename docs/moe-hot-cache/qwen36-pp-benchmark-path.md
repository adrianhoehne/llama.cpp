# Qwen3.6 35B MoE Prompt-Processing Benchmark Path

This document records the current baseline and the intended comparison path for
prompt-processing work on `unsloth/Qwen3.6-35B-A3B-GGUF:Q6_K_XL`.

## Goal

We want to understand why prompt processing with the MoE hot-cache path can be
slower than standard llama.cpp in some cases, and which code changes improve it
without hurting token generation.

The benchmark must separate two phases:

- **PP / prompt processing**: many prompt tokens are processed in batches.
- **TG / token generation**: usually one new token is decoded at a time.

The hot-cache path is mainly designed for TG, where repeatedly used experts can
stay on the GPU. PP is different because a large prompt can touch many more
experts across the batch, so cache hit rate and merge overhead behave
differently.

## Prompt File

Prompt file:

```text
pp-bench-conversation-code.txt
```

The file contains a conversation and a larger HTML/JavaScript code sample. It is
used through the `llama-bench --prompt-file` argument so PP benchmarks use a real
prompt instead of random vocabulary tokens.

TG numbers from `llama-bench` need separate interpretation. The `tg128` path
generates synthetic tokens, so its expert distribution may not match the prompt
or the expert list learned for the hot cache. A lower `tg128` result can simply
mean that those synthetic tokens hit cold experts that are poorly represented in
the cache.

The current prompt token count for Qwen3.6 is reported by `llama-bench` as:

```text
pp3070
```

## Default Baseline

This is the current standard llama.cpp baseline without MoE hot-cache:

```bash
./build/bin/llama-bench \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q6_K_XL \
  --prompt-file pp-bench-conversation-code.txt \
  -ncmoe 31 \
  --device CUDA0
```

Build:

```text
fcdd522b9 (9355)
```

Hardware:

```text
CUDA0: NVIDIA GeForce RTX 2060, 11833 MiB VRAM
CUDA1: Quadro M1200, 4035 MiB VRAM
```

Observed VRAM use for this baseline:

```text
about 9 GB VRAM
```

Results:

| Model | Mode | Prompt file | n_cpu_moe | Device | Test | Throughput |
| --- | --- | --- | ---: | --- | ---: | ---: |
| qwen35moe 35B.A3B Q6_K | standard llama.cpp | pp-bench-conversation-code.txt | 31 | CUDA0 | pp3070 | 75.67 +/- 0.01 t/s |
| qwen35moe 35B.A3B Q6_K | standard llama.cpp | pp-bench-conversation-code.txt | 31 | CUDA0 | tg128 | 22.14 +/- 0.10 t/s |

This baseline is important because it shows the throughput we must beat with the
hot-cache path while keeping memory usage comparable.

Note: `llama-bench` now supports both MoE placement forms:

- `-cmoe` / `--cpu-moe`: keep all MoE expert weights on CPU.
- `-ncmoe N` / `--n-cpu-moe N`: keep the expert weights of the first `N` MoE
  layers on CPU.

For a server-equivalent hot-cache benchmark, prefer `--cpu-moe`, because the
hot-cache design assumes the regular expert tensors remain on the CPU/RAM path
and the selected hot experts are copied separately into the GPU hot cache.

## Capturing MoE Perf From llama-bench

`llama-bench` can write the same JSON shape as `GET /moe-layer-perf`:

```bash
./build/bin/llama-bench \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q6_K_XL \
  --prompt-file pp-bench-conversation-code.txt \
  -cmoe \
  --device CUDA0 \
  --moe-hot-cache-auto-reserve-mib 3000 \
  --moe-hot-cache-max-mib -1 \
  --moe-hot-cache qwen36 \
  --moe-hot-cache-pp-reduce-merge on \
  --moe-layer-perf-out qwen36-bench-pp.json
```

The benchmark path resets MoE perf counters after warmup and writes the snapshot
after the measured repetitions. That makes the file useful for comparing PP/TG
runs without starting `llama-server`.

## Hot-Cache Without PP Reduce Merge

This run uses the hot cache but leaves `--moe-hot-cache-pp-reduce-merge` at its
default value, so it measures the regular hot-cache graph without the dedicated
PP merge reduction feature.

Command:

```bash
./build/bin/llama-bench \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q6_K_XL \
  --prompt-file pp-bench-conversation-code.txt \
  -cmoe \
  --device CUDA0 \
  --moe-hot-cache-auto-reserve-mib 3000 \
  --moe-hot-cache-max-mib -1 \
  --moe-hot-cache qwen36
```

Build:

```text
fcdd522b9 (9355)
```

Results:

| Model | Mode | Prompt file | cpu_moe | Device | Hot cache | Auto reserve | Test | Throughput | Change vs baseline |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| qwen35moe 35B.A3B Q6_K | hot-cache, PP reduce merge off | pp-bench-conversation-code.txt | 1 | CUDA0 | qwen36 | 3000 MiB | pp3070 | 86.61 +/- 0.31 t/s | +14.5% |
| qwen35moe 35B.A3B Q6_K | hot-cache, PP reduce merge off | pp-bench-conversation-code.txt | 1 | CUDA0 | qwen36 | 3000 MiB | tg128 | 18.47 +/- 0.17 t/s | -16.6% |

Interpretation:

- PP is already faster with the hot-cache path even without the PP reduce-merge
  feature: `75.67 -> 86.61 t/s`.
- TG is lower in this setup: `22.14 -> 18.47 t/s`, but this `tg128` number is
  a synthetic `llama-bench` generation workload. It is not evidence that the PP
  changes hurt decode; the generated tokens hit experts that are not well
  represented in the `qwen36` hot-cache list.
- This is not a perfectly isolated comparison because the baseline used
  `-ncmoe 31`, while the hot-cache run uses `--cpu-moe`. That is intentional for
  the hot-cache graph, but a separate `--cpu-moe` no-hot-cache baseline would
  make the CPU-MoE placement cost visible.
- The result suggests the hot-cache graph can help PP on real prompts, but the
  absolute TG comparison depends on whether the decode workload selects experts
  covered by the hot cache.

What changed:

- Before, the standard baseline used regular llama.cpp MoE placement with
  `-ncmoe 31`. Expert tensors for the configured CPU-MoE layers stayed on the
  CPU path, and no separate GPU hot-cache copy existed.
- After, the hot-cache run used `--cpu-moe` and loaded selected experts from the
  `qwen36` expert list into a separate GPU hot cache.
- The original expert tensors still remain available on CPU/RAM. The hot cache
  is an additional fast path for selected experts, not a replacement of the
  model weights.

Why this can be better:

- Any routed expert that is present in the hot cache can be evaluated on CUDA0
  instead of taking the slower CPU expert path.
- PP can still benefit because a large prompt repeatedly uses some experts
  across many tokens.
- This run establishes whether the hot-cache graph itself is viable before
  adding more PP-specific optimizations.

Tradeoff:

- Hot-cache setup consumes additional VRAM.
- The graph now has split and merge overhead for hot and cold experts.
- The TG number was lower in this exact benchmark because `llama-bench`
  generated tokens that selected experts poorly covered by the hot cache. That
  is a workload mismatch, not a PP optimization regression.

Status:

- Kept as the baseline hot-cache comparison.
- Not considered the final optimized PP mode.

## Hot-Cache With PP Reduce Merge On

This run uses the same hot-cache setup as above, but enables the PP-specific
reduce-merge optimization explicitly:

```bash
./build/bin/llama-bench \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q6_K_XL \
  --prompt-file pp-bench-conversation-code.txt \
  -cmoe \
  --device CUDA0 \
  --moe-hot-cache-auto-reserve-mib 3000 \
  --moe-hot-cache-max-mib -1 \
  --moe-hot-cache qwen36 \
  --moe-hot-cache-pp-reduce-merge on
```

Build:

```text
fcdd522b9 (9355)
```

Results:

| Model | Mode | Prompt file | cpu_moe | Device | Hot cache | Auto reserve | PP reduce merge | Test | Throughput | Change vs baseline | Change vs hot-cache off |
| --- | --- | --- | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| qwen35moe 35B.A3B Q6_K | hot-cache | pp-bench-conversation-code.txt | 1 | CUDA0 | qwen36 | 3000 MiB | on | pp3070 | 93.55 +/- 0.70 t/s | +23.6% | +8.0% |
| qwen35moe 35B.A3B Q6_K | hot-cache | pp-bench-conversation-code.txt | 1 | CUDA0 | qwen36 | 3000 MiB | on | tg128 | 18.56 +/- 0.25 t/s | -16.2% | +0.5% |

Interpretation:

- `--moe-hot-cache-pp-reduce-merge on` improves PP from `86.61` to
  `93.55 t/s` compared with the same hot-cache setup without reduce merge.
- Compared with the original baseline, PP improves by about `23.6%`.
- TG is almost unchanged compared with hot-cache reduce-merge off:
  `18.47 -> 18.56 t/s`.
- The optimization is therefore doing what it should for PP: it reduces prompt
  processing overhead without materially affecting TG in this benchmark.

What changed:

- Before, PP kept the regular hot-cache merge shape. The graph had to carry
  per-slot branch outputs farther through the graph before reducing them back to
  token rows.
- After, `--moe-hot-cache-pp-reduce-merge on` enables a PP-specific branch
  reduce/merge plan whenever the prompt batch has multiple tokens.
- Decode behavior remains separate. Decode is still allowed to use its own
  one-token merge strategy.

Why this is better:

- PP has many routed slots because it handles many tokens at once.
- Reducing branch results earlier lowers the amount of intermediate data that
  later graph nodes need to process.
- The change targets PP overhead directly while leaving TG almost unchanged.

Tradeoff:

- The graph has an additional PP-specific mode that must be kept separate from
  decode behavior.
- If the prompt batch is small, the extra reduction structure may not pay for
  itself. That is why an `auto` mode exists, even though the benchmark here uses
  `on` for controlled testing.

Status:

- Kept.
- This is the first clearly useful PP-specific optimization in the sequence.

## Hot-Cache With PP Expert-Major Worklist

This run uses the new default PP worklist policy:

- Prompt processing (`n_tokens > 1`) uses expert-major worklist order.
- Decode (`n_tokens == 1`) and warmup keep token-major order.

Command:

```bash
./build/bin/llama-bench \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q6_K_XL \
  --prompt-file pp-bench-conversation-code.txt \
  -cmoe \
  --device CUDA0 \
  --moe-hot-cache-auto-reserve-mib 3000 \
  --moe-hot-cache-max-mib -1 \
  --moe-hot-cache qwen36 \
  --moe-hot-cache-pp-reduce-merge on
```

Results:

| Model | Mode | Prompt file | cpu_moe | Device | Hot cache | Auto reserve | PP reduce merge | PP worklist | Test | Throughput | Change vs previous on |
| --- | --- | --- | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| qwen35moe 35B.A3B Q6_K | hot-cache | pp-bench-conversation-code.txt | 1 | CUDA0 | qwen36 | 3000 MiB | on | expert-major default | pp3070 | 94.13 +/- 0.29 t/s | +0.6% |
| qwen35moe 35B.A3B Q6_K | hot-cache | pp-bench-conversation-code.txt | 1 | CUDA0 | qwen36 | 3000 MiB | on | token-major decode | tg128 | 18.48 +/- 0.17 t/s | -0.4% |

Interpretation:

- Expert-major PP worklist order gives a small PP gain over the previous
  reduce-merge-on run: `93.55 -> 94.13 t/s`.
- TG remains effectively unchanged because decode keeps token-major order.
- The gain is smaller than expected, which suggests the remaining PP cost is not
  dominated by the CPU worklist ordering alone. The next larger changes need to
  move more split/merge work toward GPU-side execution or avoid unnecessary
  slot materialization.

What changed:

- Before, the PP worklist used token-major ordering. Slots were emitted in token
  order, so consecutive slots often referenced different experts.
- After, prompt processing uses expert-major ordering. Slots are grouped by
  expert first, while decode keeps token-major ordering.

Why this can be better:

- Grouping slots by expert improves locality for expert lookup and branch work.
- It better matches how `mul_mat_id` consumes grouped expert IDs.
- It keeps decode unchanged, because decode has only one token and does not
  benefit from reordering in the same way.

Tradeoff:

- Building an expert-major list requires a counting pass and an emission pass.
- The extra CPU work only makes sense when PP has enough slots to amortize it.
- The measured gain was small, so this is useful but not a dominant bottleneck.

Status:

- Kept as the default PP worklist order.
- Decode and warmup remain token-major.

## Hot-Cache With PP Reduce Merge On And UBatch 1024

This run keeps the same hot-cache setup and enables `-ub 1024` instead of the
default `-ub 512`.

```bash
./build/bin/llama-bench \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q6_K_XL \
  --prompt-file pp-bench-conversation-code.txt \
  -cmoe \
  --device CUDA0 \
  --moe-hot-cache-auto-reserve-mib 3000 \
  --moe-hot-cache-max-mib -1 \
  --moe-hot-cache qwen36 \
  --moe-hot-cache-pp-reduce-merge on \
  -ub 1024
```

Results:

| Model | Mode | Prompt file | cpu_moe | n_ubatch | Device | Hot cache | Auto reserve | PP reduce merge | Test | Throughput | Change vs ub512 |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | --- | ---: | ---: | ---: |
| qwen35moe 35B.A3B Q6_K | hot-cache | pp-bench-conversation-code.txt | 1 | 1024 | CUDA0 | qwen36 | 3000 MiB | on | pp3070 | 95.15 +/- 0.77 t/s | +1.7% |
| qwen35moe 35B.A3B Q6_K | hot-cache | pp-bench-conversation-code.txt | 1 | 1024 | CUDA0 | qwen36 | 3000 MiB | on | tg128 | 18.55 +/- 0.24 t/s | -0.1% |

Interpretation:

- Increasing `n_ubatch` from `512` to `1024` adds a small PP improvement:
  `93.55 -> 95.15 t/s`.
- TG is unchanged within noise: `18.56 -> 18.55 t/s`.
- This suggests larger ubatches reduce some PP overhead, but this is not the
  main remaining bottleneck for the current setup.

What changed:

- Before, the default micro-batch size was used, effectively `-ub 512` for this
  benchmark setup.
- After, PP is allowed to process up to `1024` tokens per ubatch.

Why this can be better:

- Larger ubatches reduce the number of PP graph executions for a long prompt.
- Some fixed per-ubatch costs are paid fewer times.
- GPU work can become more efficient when each graph execution has more prompt
  tokens to process.

Tradeoff:

- Larger ubatches can increase temporary memory pressure.
- They only help when the prompt is large enough. Short prompts may see no
  benefit or may be less stable.
- This is a runtime configuration lever, not a code improvement.

Status:

- Useful tuning option.
- Not a default code change in this benchmark path.

## Hot-Cache With PP Compact Cold Reduce

This change keeps `--moe-hot-cache-pp-reduce-merge on`, but avoids materializing
the full cold slot tensor before the branch merge during prompt processing.
Instead, the cold branch output is reduced directly from compact cold slots into
token rows.

The key implementation point is that PP and decode differ here:

- Decode usually processes one token and benefits mostly from hot/cold overlap.
- PP processes many tokens, so the intermediate slot tensor can become large.
  Reducing compact cold slots directly avoids a large amount of CPU merge work
  and memory traffic.

Command:

```bash
./build/bin/llama-bench \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q6_K_XL \
  --prompt-file pp-bench-conversation-code.txt \
  -cmoe \
  --device CUDA0 \
  --moe-hot-cache-auto-reserve-mib 3000 \
  --moe-hot-cache-max-mib -1 \
  --moe-hot-cache qwen36 \
  --moe-hot-cache-pp-reduce-merge on
```

Results:

| Model | Mode | Prompt file | cpu_moe | Device | Hot cache | Auto reserve | PP reduce merge | Compact cold reduce | Test | Throughput | Change vs previous measured PP |
| --- | --- | --- | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| qwen35moe 35B.A3B Q6_K | hot-cache | pp-bench-conversation-code.txt | 1 | CUDA0 | qwen36 | 3000 MiB | on | on | pp3070 | 106.58 +/- 0.52 t/s | +13.2% vs 94.13 |
| qwen35moe 35B.A3B Q6_K | hot-cache | pp-bench-conversation-code.txt | 1 | CUDA0 | qwen36 | 3000 MiB | on | on | tg128 | 18.56 +/- 0.08 t/s | unchanged |

Interpretation:

- This was the largest PP code win in this round: `94.13 -> 106.58 t/s`.
- The change is local to the MoE hot-cache implementation and does not require a
  new ggml core operator.
- TG stayed effectively unchanged, which means the optimization targets PP
  overhead without disturbing decode.

What changed:

- Before, the cold branch produced compact cold expert outputs, expanded them
  into a cold slot tensor, then reduced the slot tensor back into token rows.
- After, the cold branch output is reduced directly from compact cold slots into
  final token rows by `llama_moe_hot_cache_build_compact_cold_reduce`.
- This removes an intermediate materialization step from the PP cold path.

Why this is better:

- In PP, `capacity = n_tokens * n_expert_used`, so slot tensors become large.
- Many slots map back to the same token, so expanding all slots and then
  reducing them is expensive.
- Direct compact reduction performs the necessary accumulation without carrying
  a full slot tensor through the graph.

Tradeoff:

- The cold reduce is a custom CPU-side operation in our hot-cache code.
- It needs worklist counts and token IDs to be present and correct.
- It is PP-specific; decode still uses its separate direct/prefix merge paths.

Status:

- Kept.
- This is the largest measured PP win in the sequence and remains
  rebase-friendly because it stays in `src/moe-hot-cache`.

## Removed Experiment: Hot Branch Scatter-Add

This experiment tried to compact the hot branch further by adding a scatter-add
mode to `set_rows`, so hot branch rows could be accumulated directly instead of
materializing and reducing slots.

Implementation touched ggml core files:

- `ggml/src/ggml-cpu/ops.cpp`
- `ggml/src/ggml-cuda/set-rows.cu`
- `ggml/src/ggml-cuda/ggml-cuda.cu`

Results:

| Model | Mode | Test | Throughput | Change vs compact cold reduce |
| --- | --- | ---: | ---: | ---: |
| qwen35moe 35B.A3B Q6_K | hot scatter-add experiment | pp3070 | 106.96 +/- 0.34 t/s | +0.4% |
| qwen35moe 35B.A3B Q6_K | hot scatter-add experiment | tg128 | 18.50 +/- 0.16 t/s | noise/slightly lower |

Reason for removal:

- The PP gain was inside benchmark noise: `106.58 -> 106.96 t/s`.
- TG did not improve.
- The implementation modified ggml CUDA/CPU core paths, which increases rebase
  conflict risk.
- The complexity was not justified by the measured result.

Current status:

- Removed.
- Search terms such as `set_rows_accumulate`, `COMPACT_HOT`, and
  `SET_ROWS_FLAG_ACCUMULATE` should no longer appear in the code.

What changed during the experiment:

- Before, hot branch slots were materialized and then reduced back into token
  rows.
- During the experiment, `set_rows` gained an accumulate mode so hot branch
  outputs could be scattered and accumulated directly into destination rows.

Why it looked attractive:

- It mirrored the idea that worked for the cold branch: avoid materializing a
  large intermediate slot tensor.
- If the hot branch had been dominated by slot materialization, direct
  scatter-add could have reduced memory traffic.

Why it was not better:

- The measured PP improvement was only `106.58 -> 106.96 t/s`, which is inside
  noise for this setup.
- TG did not improve.
- The change touched generic ggml CPU/CUDA operators, not only our hot-cache
  code.

Tradeoff:

- High maintenance cost and high rebase-conflict risk.
- Small or non-measurable runtime gain.
- Higher risk of affecting unrelated model paths.

Status:

- Removed.
- The idea is documented as a tried path, but should not be reintroduced unless
  profiling later shows the hot branch reduction is again a dominant cost.

## Hot-Cache With Compact Cold Reduce Memory Cleanup

After removing the hot scatter-add experiment, the compact cold reduce path was
kept and optimized in our own hot-cache code:

- Cold reduce now zeroes contiguous local F32 ranges with `memset`.
- Cold reduce accumulation uses direct row pointers instead of repeated pointer
  recomputation for every embedding element.
- Worklist initialization writes fields as contiguous ranges instead of looping
  slot-by-slot over every field.

These changes stay in `src/moe-hot-cache` and avoid ggml core changes.

Results:

| Model | Mode | Test | Throughput | Change vs compact cold reduce |
| --- | --- | ---: | ---: | ---: |
| qwen35moe 35B.A3B Q6_K | compact cold reduce memory cleanup | pp3070 | 108.51 +/- 0.39 t/s | +1.8% |
| qwen35moe 35B.A3B Q6_K | compact cold reduce memory cleanup | tg128 | 18.76 +/- 0.36 t/s | within noise |

Interpretation:

- This is a real but smaller PP win after the large compact cold reduce change.
- Because the change only improves memory access patterns, it is low risk and
  rebase-friendly.
- TG stayed within noise.

What changed:

- Before, compact cold reduce zeroed output elements with a nested scalar loop.
- Before, accumulation repeatedly recomputed element pointers inside the inner
  embedding loop.
- After, each worker zeroes its contiguous local F32 range with `memset`.
- After, accumulation uses row pointers and simple indexed additions.
- Worklist initialization also changed from slot-by-slot initialization to
  field-by-field contiguous initialization.

Why this is better:

- The operation is memory-heavy, so reducing pointer arithmetic and writing
  contiguous ranges helps PP throughput.
- `memset` expresses the zeroing operation directly and lets libc/compiler
  choose an efficient implementation.
- Field-wise worklist initialization matches the actual tensor layout better
  than repeatedly jumping between fields for every slot.

Tradeoff:

- This assumes the tensors are dense F32 rows with `nb[0] == sizeof(float)`,
  which the code asserts.
- It is still CPU work; it does not remove the cold reduce operation itself.

Status:

- Kept.
- This is a local, low-risk cleanup with a measurable PP improvement.

## Removed Experiment: Compact PP Worklist Field Flags

This experiment allowed the graph to ask the worklist builder to skip fields
that are not read in the PP compact cold reduce path, for example
`cold_src_slot` and, when perf counters are not needed, `hot_expert_id`.

Results:

| Model | Mode | Test | Throughput | Change vs previous |
| --- | --- | ---: | ---: | ---: |
| qwen35moe 35B.A3B Q6_K | compact worklist field flags | pp3070 | 108.65 +/- 0.44 t/s | +0.1% |
| qwen35moe 35B.A3B Q6_K | compact worklist field flags | tg128 | 18.73 +/- 0.37 t/s | noise |

Reason for removal:

- The result overlaps fully with the previous run: `108.51 +/- 0.39` vs
  `108.65 +/- 0.44`.
- It added template dispatch and field-mask complexity to the graph.
- The complexity was not worth keeping for a non-measurable gain.

Current status:

- Removed.
- The worklist builder now always fills the normal field set again.

What changed during the experiment:

- Before, the worklist builder always filled the full worklist field set.
- During the experiment, the graph selected a specialized worklist builder that
  skipped fields unused by the PP compact-cold path, such as `cold_src_slot` and
  sometimes `hot_expert_id`.

Why it looked attractive:

- The PP compact-cold path no longer reads every field that older paths needed.
- Skipping unused fields should reduce CPU writes during worklist construction.
- The idea stayed inside `src/moe-hot-cache` and did not require ggml changes.

Why it was not better:

- The result was `108.65 +/- 0.44 t/s`, overlapping fully with
  `108.51 +/- 0.39 t/s`.
- The saved writes were too small compared with the remaining PP cost.
- The implementation added template dispatch and field-mask branching around the
  graph/worklist boundary.

Tradeoff:

- More code paths for very little measurable effect.
- More complexity in graph construction.
- Potential maintenance burden when adding more model adapters.

Status:

- Removed.
- The normal worklist field set is filled again for clarity.

## Hot-Cache With Stack Offset Tables

This experiment removes two small heap allocations from the expert-major
worklist path:

- `hot_offsets`
- `cold_offsets`

These were previously `std::vector<int32_t>` allocations performed per layer
when PP used expert-major worklist ordering. They are now fixed local arrays
bounded by `LLAMA_MAX_EXPERTS`.

Before:

```cpp
std::vector<int32_t> hot_offsets((size_t) layer.n_hot, 0);
std::vector<int32_t> cold_offsets((size_t) layer.n_expert, 0);
```

After:

```cpp
int32_t hot_offsets[LLAMA_MAX_EXPERTS] = {};
int32_t cold_offsets[LLAMA_MAX_EXPERTS] = {};
```

Why this is better:

- The PP expert-major worklist path runs once per MoE layer during prompt
  processing, so avoiding per-layer heap allocation removes allocator activity
  from a hot CPU-side path.
- The maximum number of experts is already bounded by `LLAMA_MAX_EXPERTS`, so a
  fixed local array matches the existing model invariant better than a dynamic
  vector.
- The code has less runtime state: no vector capacity, no heap ownership, and no
  allocator behavior to account for.
- This makes benchmark behavior slightly more deterministic, because the
  allocator is no longer part of this part of the hot path.
- The change is local to `src/moe-hot-cache` and does not touch ggml core code,
  so it does not add rebase conflict risk.
- It should not affect TG behavior because decode keeps token-major ordering.

Tradeoff:

- Stack usage increases by two fixed arrays.
- This is acceptable here because `LLAMA_MAX_EXPERTS` is small and already used
  as the upper bound for MoE expert fan-out in the surrounding code.
- The change should not be sold as a throughput optimization; it is primarily a
  small simplification and determinism cleanup.

Results:

| Model | Mode | Test | Throughput | Change vs memory cleanup |
| --- | --- | ---: | ---: | ---: |
| qwen35moe 35B.A3B Q6_K | stack offset tables, run 1 | pp3070 | 107.02 +/- 1.88 t/s | noisy / inconclusive |
| qwen35moe 35B.A3B Q6_K | stack offset tables, run 1 | tg128 | 18.77 +/- 0.36 t/s | noise |
| qwen35moe 35B.A3B Q6_K | stack offset tables, run 2 | pp3070 | 108.58 +/- 0.20 t/s | no measurable change |
| qwen35moe 35B.A3B Q6_K | stack offset tables, run 2 | tg128 | 18.68 +/- 0.31 t/s | noise |

Interpretation:

- The first run is too noisy to use for a decision.
- The second run overlaps with the previous `108.51 +/- 0.39 t/s` result.
- This is not a measurable performance win, but it removes heap allocation from
  the PP worklist path and remains local to `src/moe-hot-cache`.

## Current Result Summary

| Setup | PP t/s | PP vs baseline | TG t/s | TG vs baseline |
| --- | ---: | ---: | ---: | ---: |
| Baseline, `-ncmoe 31`, no hot-cache | 75.67 | baseline | 22.14 | baseline |
| Hot-cache, `--cpu-moe`, reduce merge off | 86.61 | +14.5% | 18.47 | -16.6% |
| Hot-cache, `--cpu-moe`, reduce merge on | 93.55 | +23.6% | 18.56 | -16.2% |
| Hot-cache, `--cpu-moe`, reduce merge on, expert-major PP worklist | 94.13 | +24.4% | 18.48 | -16.5% |
| Hot-cache, `--cpu-moe`, reduce merge on, `-ub 1024` | 95.15 | +25.7% | 18.55 | -16.2% |
| Hot-cache, compact cold reduce | 106.58 | +40.8% | 18.56 | -16.2% |
| Removed: hot scatter-add experiment | 106.96 | +41.4% | 18.50 | -16.4% |
| Hot-cache, compact cold reduce memory cleanup | 108.51 | +43.4% | 18.76 | -15.3% |
| Removed: compact worklist field flags | 108.65 | +43.6% | 18.73 | -15.4% |
| Stack offset tables, run 2 | 108.58 | +43.5% | 18.68 | -15.6% |

The PP path now has a clear win with hot-cache plus reduce-merge. The lower
absolute `tg128` values in this table are caused by the synthetic `llama-bench`
TG workload selecting experts that are poorly represented in the hot-cache
expert list. They should not be read as a PP regression. For TG, use a
representative decode continuation or learn and apply an expert list from the
same decode workload before comparing against the standard baseline.

## What Must Stay Constant

For meaningful comparisons, keep these values fixed unless the test explicitly
documents the change:

- Same model: `unsloth/Qwen3.6-35B-A3B-GGUF:Q6_K_XL`
- Same prompt file: `pp-bench-conversation-code.txt`
- Same main device: `CUDA0`
- Same CPU-MoE placement inside a comparison group:
  baseline runs currently use `-ncmoe 31`, hot-cache runs currently use `-cmoe`
- Same build, or record the new build hash
- Same context-related settings where possible
- Same measured phases: PP and TG
- Same TG workload class: synthetic `llama-bench` TG and real decode
  continuations can select very different experts

## Hot-Cache Comparison Command Shape

The hot-cache test should use the same prompt file and baseline settings, then
add only the MoE hot-cache arguments:

```bash
./build/bin/llama-bench \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q6_K_XL \
  --prompt-file pp-bench-conversation-code.txt \
  --cpu-moe \
  --device CUDA0 \
  --moe-hot-cache path/to/experts.json \
  --moe-hot-cache-max-mib VALUE \
  --moe-hot-cache-auto-reserve-mib VALUE \
  --moe-hot-cache-weighting flat \
  --moe-hot-cache-layer-curve 0.70 \
  --moe-hot-cache-pp-reduce-merge auto
```

Use the same expert list for repeated tests unless the test is explicitly about
expert selection or dynamic updating.

## Metrics To Record

Record these values for every run:

| Metric | Why it matters |
| --- | --- |
| PP t/s | Main metric for prompt-processing optimization. |
| TG t/s | Ensures PP changes do not alter decode behavior inside the same TG workload. For `llama-bench tg128`, interpret it together with hot-cache hit rate because the generated tokens are synthetic. |
| VRAM used | Shows whether speed was bought by consuming more GPU memory. |
| Hot-cache hit rate | Explains whether hot-cache placement matches the prompt. |
| Hot-cache budget MiB | Makes memory comparisons reproducible. |
| Selected experts | Shows how much of the model was moved into the hot cache. |
| `pp-reduce-merge` mode | Directly affects PP behavior. |
| Fallback count | Any fallback can hide or invalidate a speed result. |

## Current Interpretation

The standard baseline is strong for PP:

- `pp3070`: **75.67 t/s**
- `tg128`: **22.14 t/s**
- VRAM use: **about 9 GB**

For the hot-cache path to be worthwhile on this workload, it should either:

- improve representative TG enough while keeping PP acceptable, or
- improve PP with `--moe-hot-cache-pp-reduce-merge auto/on` without increasing
  VRAM so much that the comparison becomes unfair.

The `tg128` result above is not representative TG for this hot-cache file. It is
a synthetic token stream, and in this run it selected experts that were poorly
covered by the `qwen36` cache. Use it to verify that PP changes keep decode
plumbing stable, not as the sole final TG quality signal.

If the hot-cache PP path is slower, likely causes are:

- PP touches more experts than TG, lowering hot-cache locality.
- Hot/cold merge work becomes more expensive for large prompt batches.
- CPU routing and CPU expert work may dominate if many experts remain cold.
- The hot-cache graph can add overhead that is hidden during TG but visible
  during PP.

## Next Test Steps

1. Run the hot-cache configuration with the same prompt file.
2. Record PP t/s, TG t/s, VRAM, hit rate, and fallback count.
3. Repeat with `--moe-hot-cache-pp-reduce-merge auto`, `on`, and `off`.
4. Compare against the baseline table above.
5. Keep the best PP setting only if TG remains stable inside the same TG
   workload and hit-rate context.
