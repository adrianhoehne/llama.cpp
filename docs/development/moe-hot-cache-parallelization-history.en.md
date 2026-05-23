# MoE Hot-Cache Parallelization: Changes, Reasons, And Impact

State of this note: analysis of `0641adb9e08da0a675058bc39a8c928a7f8d6ad0..HEAD`
on branch `cached-experts-v2` with `HEAD = 1618089c0`, plus the then-uncommitted
change in `src/models/qwen35moe.cpp`.

## Starting Point And Goal

The starting point was a llama.cpp state before the hot-cache work. The goal was
to speed up Qwen3.5/Qwen3.6 MoE decode on one GPU plus CPU, even though not all
experts fit sensibly on the GPU at the same time.

The basic idea:

- copy frequently used experts per layer into a GPU hot cache,
- leave less frequently used experts on CPU,
- split the MoE selection into hot and cold work for each decode step,
- run the GPU hot lane and CPU cold lane in parallel,
- merge both results back into the normal layer output afterward.

The rough benchmark from the runs was:

- without hot/cold parallelization on a simple "Hallo" prompt: about `19.7 tk/s`,
- first hot/cold variant: simple prompts about `13.96 tk/s`, programming prompt
  about `16.4-17 tk/s`,
- current state with `performance56.json`: about `28.09 tk/s`.

These values are not all strict A/B measurements because prompt, run length,
hot-cache JSON, perf counters, and individual toggles changed over the
iterations. The direction is still clear: the first working parallelization
became a much faster decode path.

## Affected Areas

Since the start commit, the relevant work changed:

- `src/llama-moe-hot-cache.{h,cpp}`
- `src/models/qwen35moe.cpp`
- `src/llama-context.cpp`
- `src/llama-graph.{h,cpp}`
- `src/llama-model.{h,cpp}`
- `src/models/models.h`
- `ggml/include/ggml-backend.h`
- `ggml/include/ggml.h`
- `ggml/src/ggml-backend.cpp`
- `ggml/src/ggml-cpu/ggml-cpu.c`
- `ggml/src/ggml-cpu/ops.cpp`
- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `ggml/src/ggml-cuda/reduce_rows.cuh`
- `ggml/src/ggml-cuda/sumrows.cu`
- `tests/test-moe-hot-cache.cpp`

Local helper files were also used for tests and server starts, including
`model_config.ini`, `start-server`, and `start-server-performance14`. At the
time this note was created, these files were still untracked and therefore not
part of `HEAD`.

## Commit Timeline

The main commits since `0641adb9e08da0a675058bc39a8c928a7f8d6ad0`:

- `5ad16adeb`: first inference test with hot-cache integration. Correctness was
  more important than speed; throughput dropped as expected.
- `c26ec8556`: runnable PoC. The hot-cache path became executable and got first
  tests.
- `61aaa4c75`: more MoE performance metrics to make visible where time is lost.
- `338c5ee05`: first parallelization attempt, still broken.
- `2a795fc3a`: revert of the broken parallelization attempt.
- `980ad5292`: removal of individual meta documents in the experimental fork.
  This had no runtime effect.
- `dac2548ac`: first working hot/cold parallelization. CPU and GPU were computed
  separately and timed in parallel.
- `39a942ab1`: scheduler optimization.
- `50284e9b8`: further speed work on the parallel path.
- `220b44da3`: new method for selecting hot experts from perf data.
- `5759b13aa`: adjusted selection weighting.
- `03b23a2f6`: improved routing.
- `9987d8266`: reduced the perf path to data relevant for optimization.
- `19c309c84`: made perf counters cleanly disableable via `--no-perf`.
- `d7305c2d0`: further performance work, including merge/sum path work.
- `a47d68214`: faster cold path.
- `1618089c0`: current committed state at about `28 tk/s`.

After that, there was still an uncommitted change in `src/models/qwen35moe.cpp`:
`LLAMA_MOE_HOT_CACHE_COLD_FIRST_ROW_INPUT`. In `performance56.json`, this
reduced the worklist/cold-gather share, but shifted some time into merge,
cold-lane, and join-wait.

## Hot-Cache Data Model And Expert Loading

### What Changed

`src/llama-moe-hot-cache.{h,cpp}` introduces a dedicated data model for the
MoE hot cache:

- `llama_moe_hot_cache_entry`
- `llama_moe_hot_cache_expert_size`
- `llama_moe_hot_cache_plan`
- `llama_moe_hot_cache_layer`
- `llama_moe_hot_cache`

Initialization reads a perf JSON file, scores experts per layer, plans the
selection within a MiB budget, and copies the selected expert tensors into
GPU-side hot-cache tensors. The Qwen MoE FFN tensors for gate, up, down, and the
fused gate-up case are supported.

Model-level parameters were added for path and budget:

- `--moe-hot-cache <json>`
- `--moe-hot-cache-max-mib <mib>`

### Why

Keeping all experts on the GPU is not the useful path for the target model. At
the same time, decode is highly repetitive: many layers and prompts use specific
experts much more frequently. The cache therefore spends GPU memory on experts
that avoid the most cold-path work.

### Impact

The hot cache is the foundation for the whole speedup. In later runs, the hot
slot ratio stabilized at about `68-69%`. Example from `performance56.json`:

- `n_expert = 256`
- `n_expert_used = 8`
- `hot_slot_ratio = 0.683649`

That leaves about 31-32% of MoE slots in the cold path. This fits the assumption
that the reachable GPU hit rate is probably capped around 70%.

## Qwen3.5-MoE Graph: Hot/Cold Split

### What Changed

For active hot-cache layers, `src/models/qwen35moe.cpp` no longer builds the
normal MoE FFN path, but a specialized hot/cold path:

1. Compute router logits.
2. Build a worklist.
3. Split the worklist into hot IDs, cold IDs, weights, token/slot IDs, and counts.
4. Compute the hot branch with cached experts.
5. Compute the cold branch with normal layer experts.
6. Merge the results.

The worklist fields are:

- `HOT_ID`
- `HOT_SRC_SLOT`
- `HOT_TOKEN_ID`
- `HOT_WEIGHT`
- `COLD_ID`
- `COLD_SRC_SLOT`
- `COLD_TOKEN_ID`
- `COLD_WEIGHT`
- `HOT_EXPERT_ID`
- `HOT_COUNT`
- `COLD_COUNT`

The shared FFN helper was extended so it accepts external expert IDs, compact
weights, and branch-specific backends. This allows the hot branch to explicitly
use CUDA and the cold branch to explicitly use CPU.

### Why

The normal MoE path computes one shared expert selection. Parallel execution
requires two independent work sets:

- Hot: experts that are present in the GPU cache.
- Cold: all remaining experts.

Only once both sets are compact and explicitly marked can the scheduler treat
them as a fork/join region.

### Impact

The graph split made hot/cold parallelization possible at all. It was also the
area where the first errors appeared:

- a `ggml_view_1d` bounds assert during the first inference,
- then a scheduler error: `region split order is not hot-then-cold-then-join`.

The solution was to build the region more explicitly, place the counts as prefix
views, and strictly validate in the scheduler that the split order is
Hot -> Cold -> Join.

## CPU Routing And Worklist Creation

### What Changed

For decode with `n_tokens == 1`, a CPU custom op was added:

- `llama_qwen35moe_hot_cache_build_worklist_from_logits_op`

This op computes Top-K, softmax/normalization, weight scaling, and hot/cold
packing directly from the router logits. For prefill or other cases, the path
still goes through:

- `ggml_argsort_top_k`
- `ggml_get_rows`
- `ggml_soft_max`
- `llama_qwen35moe_hot_cache_build_worklist_op`

The decode CPU-routing path is enabled by default and can be disabled via env:

- `LLAMA_MOE_HOT_CACHE_CPU_DECODE_ROUTING=0`

### Why

In decode, `n_tokens == 1`. The normal graph path creates a lot of small work:
argsort, get-rows, softmax, views, and casts. This work is relatively expensive
compared with the actual MoE matmul work. The many views also created fragile
graph edges early on.

### Impact

Routing and worklist creation became more stable and measurably smaller. In the
current metrics, routing is still a large block, but later optimizations reduced
worklist time substantially:

- `performance45.json`: `worklist_time_per_call_us = 144.66`
- `performance52.json`: `worklist_time_per_call_us = 113.361`
- `performance56.json`: `worklist_time_per_call_us = 65.3677`

## Scheduler: Parallel Hot/Cold Region

### What Changed

An experimental scheduler annotation was introduced in `ggml/include/ggml-backend.h`
and `ggml/src/ggml-backend.cpp`:

- `ggml_backend_sched_moe_hot_cache_parallel_region(...)`
- `ggml_backend_sched_set_moe_hot_cache_parallel_perf_enabled(...)`
- `ggml_backend_sched_get_moe_hot_cache_parallel_perf(...)`

The scheduler uses this to identify a fork/join region:

- hot start to hot end on CUDA,
- cold start to cold end on CPU,
- join node afterward.

The cold lane is executed through a CPU worker thread in parallel with the hot
lane. The thread itself runs on the CPU; GPU work only happens when the hot lane
submits CUDA kernels to the CUDA backend.

Parallel mode is controlled through:

- `LLAMA_MOE_HOT_CACHE_PARALLEL=0`: off
- `LLAMA_MOE_HOT_CACHE_PARALLEL=1`: auto
- `LLAMA_MOE_HOT_CACHE_PARALLEL=force`: force
- `LLAMA_MOE_HOT_CACHE_PARALLEL_MIN_SLOTS`: minimum number of slots

Important: in code, parallel is now auto without an env var. The local
`start-server` still sets `LLAMA_MOE_HOT_CACHE_PARALLEL_MIN_SLOTS=0` so the
parallel region is attempted even for small decode regions.

### Why

Without the scheduler extension, llama.cpp computes backend splits serially.
Then the GPU hot cache can reduce cold work, but CPU and GPU work do not overlap
cleanly. The goal was to use otherwise idle CPU/GPU time during decode.

### Impact

This was the first large lever. After the working parallel path, the 20 tk/s
mark was reached and later exceeded substantially.

In `performance56.json`, the current state is visible:

- `parallel_region_wall_time_per_call_us = 422.892`
- `parallel_hot_lane_wall_time_per_call_us = 123.504`
- `parallel_cold_lane_wall_time_per_call_us = 382.102`
- `parallel_join_wait_time_per_call_us = 332.197`
- `parallel_fallbacks = 0`

This means the parallel region runs stably without fallbacks, but the cold lane
is still the longer branch. The join therefore often waits on the cold path.

## Backend And GGML Op Extensions

### `mul_mat_id` Flags

`src/llama-graph.h` gained new flags:

- `LLM_MUL_MAT_ID_FLAG_ALLOW_DUPLICATE_IDS`
- `LLM_MUL_MAT_ID_FLAG_ALLOW_NEGATIVE_IDS`
- `LLM_MUL_MAT_ID_FLAG_SKIP_NEGATIVE_ID_OUTPUT_ZERO`
- `LLM_MUL_MAT_ID_FLAG_SHARED_INPUT_ROW`

These flags were wired into `src/llama-graph.cpp`, the CPU backend, and the
CUDA backend.

### Why

The compact hot/cold path has different requirements from the standard MoE path:

- padding slots can contain negative IDs,
- decode uses the same input token for many expert slots,
- unused padding outputs do not always need to be zeroed if they are guaranteed
  to be ignored later,
- duplicate IDs are allowed in the compact representation.

### Impact

The flags allow smaller and cheaper branches:

- less unnecessary zeroing,
- fewer input copies,
- less work for invalid padding slots,
- foundation for `SHARED_INPUT_ROW` and later cold-path optimizations.

### `get_rows` First-Row-Only

`GGML_GET_ROWS_FLAG_FIRST_ROW_ONLY` was added in `ggml/include/ggml.h` and used
in the CPU get-rows path.

Why: in decode, the input is only one row. The cold path does not need to copy
the same activation fully again and again for multiple expert slots.

Impact: the cold gather share later dropped substantially. In `performance56.json`,
`cold_gather_scatter_time_per_call_us` is only `2.56514`, while it was still
`10.2372` in `performance55.json`.

### `sum_rows` And Strided Inputs

CPU and CUDA `sum_rows` were extended, including:

- CPU can parallelize compatible cases,
- CPU handles strided input better,
- CUDA got a strided F32 reduce path in `reduce_rows.cuh` / `sumrows.cu`.

Why: the merge path creates compact or strided representations. A forced `cont`
or summing over full capacity costs a lot.

Impact: this work was the foundation for direct merge, prefix sum, and later
merge optimizations. Merge is still the largest block.

## Decode Merge, Prefix Sum, And Graph Pruning

### Direct Decode Merge

For decode, the hot/cold path can directly produce an `[n_embd, 1]` result
instead of first building a full slot-tensor layout and summing afterward.

Switch:

- `LLAMA_MOE_HOT_CACHE_DECODE_DIRECT_MERGE`

Default: enabled.

Why: with `n_tokens == 1`, the full slot layout is only intermediate work.

Impact: fewer views, fewer `set_rows`, less sum work.

### Cold Prefix Sum

The cold path sorts valid cold slots as a prefix. Instead of summing over full
capacity including padding, only the valid prefix is processed.

Switch:

- `LLAMA_MOE_HOT_CACHE_COLD_PREFIX_SUM`

Default: enabled.

Why: padding slots do not contribute anything, but still cost time in merge and
CPU/CUDA backend paths.

### Weighted Cold Prefix Sum

The cold path can combine weighting and prefix sum in one custom op:

- `llama_qwen35moe_hot_cache_sum_weighted_prefix_rows_op`

In this mode, the cold FFN uses `apply_weights=false`; weighting happens during
the prefix merge.

Switch:

- `LLAMA_MOE_HOT_CACHE_COLD_PREFIX_WEIGHTED_SUM`

Default: enabled.

Impact in direct comparison:

- `performance52.json -> performance53.json`
- `total_moe_time_per_call_us`: `900.611 -> 884.981` (`-15.63 us`)
- `merge_time_per_call_us`: `475.775 -> 459.535` (`-16.24 us`)
- `parallel_cold_lane_wall_time_per_call_us`: `383.962 -> 369.964` (`-14.00 us`)
- `parallel_join_wait_time_per_call_us`: `327.476 -> 310.884` (`-16.59 us`)

### Graph Pruning

Later, unused worklist views were no longer built at all in the decode path:

- `hot_src_slots` only when direct merge is not active,
- `cold_src_slots` only when direct merge is not active,
- `cold_weights` not when weighted prefix sum is active,
- `hot_expert_ids` only when expert counts are really enabled.

Impact:

- `performance53.json -> performance54.json`
- `total_moe_time_per_call_us`: `884.981 -> 842.767` (`-42.214 us`)
- `worklist_time_per_call_us`: `113.256 -> 87.951` (`-25.305 us`)
- `hot_gather_scatter_time_per_call_us`: `22.807 -> 12.9825` (`-9.8245 us`)
- `cold_gather_scatter_time_per_call_us`: `19.8564 -> 10.2588` (`-9.5976 us`)

This was one of the largest late single levers.

### Repeat Hot Input

For decode, the hot input can be created with `ggml_repeat_4d(cur, capacity)`
instead of using `hot_token_ids` plus `get_rows`.

Switch:

- `LLAMA_MOE_HOT_CACHE_DECODE_REPEAT_HOT_INPUT`

Default: enabled.

Impact:

- `performance54.json -> performance55.json`
- `total_moe_time_per_call_us`: `842.767 -> 820.799` (`-21.968 us`)
- `worklist_time_per_call_us`: `87.951 -> 76.8338` (`-11.1172 us`)
- `hot_gather_scatter_time_per_call_us`: `12.9825 -> 4.95083` (`-8.03167 us`)

### Cold First-Row Input

The last uncommitted optimization builds a cold-path input matrix from the first
row during decode, instead of repeatedly copying the same row through get-rows.

Switch:

- `LLAMA_MOE_HOT_CACHE_COLD_FIRST_ROW_INPUT`

Default in the current worktree: enabled.

Impact:

- `performance55.json -> performance56.json`
- `total_moe_time_per_call_us`: `820.799 -> 816.307` (`-4.492 us`)
- `worklist_time_per_call_us`: `76.8338 -> 65.3677` (`-11.4661 us`)
- `cold_gather_scatter_time_per_call_us`: `10.2372 -> 2.56514` (`-7.67206 us`)

But:

- `merge_time_per_call_us`: `460.041 -> 471.037` (`+10.996 us`)
- `parallel_cold_lane_wall_time_per_call_us`: `369.626 -> 382.102` (`+12.476 us`)
- `parallel_join_wait_time_per_call_us`: `313.532 -> 332.197` (`+18.665 us`)

Interpretation: slightly positive overall in the measured run, but not clearly
dominant. This switch should remain A/B tested.

## Perf Data And `--no-perf`

### What Changed

`src/llama-context.cpp` got specialized MoE perf collection. The old heatmap
view was reduced for optimization and replaced with a leaner schema:

- `llama.cpp.moe_layer_opt_perf.v1`

The JSON now mainly contains:

- summary across all layers,
- hot/cold slot ratio,
- timings for routing, worklist, merge,
- hot/cold branch times,
- hot/cold gather/scatter,
- parallel region, hot lane, cold lane, join wait,
- fallback counters and fallback reasons.

Expert counts are controlled today by the MoE perf mode:

- `full`: all counters and timing fields,
- `update`: only expert/slot data plus hot/cold/join wait times,
- `off`: no MoE perf collection.

Perf collection is active only when the mode is not `off`. `--no-perf` sets the
initial mode to `off`; `POST /moe-layer-perf` can switch it back to `update` or
`full` at runtime.

### Why

The layer/expert utilization visualization was useful for the first cache plan,
but speed optimization needed different data:

- Where is the time?
- Is the cold lane longer than the hot lane?
- Is the join waiting?
- Are there scheduler fallbacks?
- Is merge or routing larger than the actual matmuls?

### Impact

The later phase of optimization was only possible in a targeted way because of
these metrics. Example from `performance56.json`:

- `total_moe_time_per_call_us = 816.307`
- `routing_time_per_call_us = 264.614`
- `worklist_time_per_call_us = 65.3677`
- `merge_time_per_call_us = 471.037`
- `parallel_region_wall_time_per_call_us = 422.892`
- `parallel_hot_lane_wall_time_per_call_us = 123.504`
- `parallel_cold_lane_wall_time_per_call_us = 382.102`
- `parallel_join_wait_time_per_call_us = 332.197`
- `parallel_fallbacks = 0`

Important: these categories are not all additive. Parallel region, branch, and
merge timings partly overlap conceptually. As diagnostics they still show
clearly that merge, cold lane, and join wait are the dominant topics now.

## Hot-Expert Selection From Perf JSON

### What Changed

The parser in `llama_moe_hot_cache_parse_perf_json` now accepts:

- `llama.cpp.moe_layer_perf.v1`
- `llama.cpp.moe_layer_opt_perf.v1`

It can read both old raw `experts` counts and new branch counts:

- `hot_experts`
- `cold_experts`

Scoring also accounts for layer weighting based on cold wait cost:

- `parallel_join_wait_time_per_call_us`
- `cold_slots_per_call`
- alternatively the difference between cold lane and hot lane

The weighting is damped and bounded so the cache does not churn heavily on every
run. Experts that were already hot receive a small sticky bonus.

### Why

A pure hit heatmap is not enough. An expert in a layer with little cold wait is
less valuable than an expert in a layer where the CPU lane slows the join. The
cache should spend GPU memory where it reduces join wait.

### Impact

The hot slot ratio did not become maximal, but it became better aligned with
expensive layers. This explains why later runs became faster despite hot slot
ratios around 68-69%, compared with earlier runs that sometimes had a higher
ratio. Example:

- `performance45.json`: `hot_slot_ratio = 0.762117`, but
  `total_moe_time_per_call_us = 944.125`
- `performance56.json`: `hot_slot_ratio = 0.683649`, but
  `total_moe_time_per_call_us = 816.307`

More hot hits alone are therefore not automatically better; what matters is
whether cold-tail time decreases.

## Start Scripts And Local Configuration

### `model_config.ini`

The local config for Qwen3.6-35B-A3B uses, among other things:

- `device = CUDA0`
- `ngl = 999`
- `cpu-moe = true`
- `ctx-size = 32000`
- `override-kv = qwen35moe.expert_used_count=int:8`

The important point is `n_expert_used = 8`. The entire optimization and measured
perf data refer to this Top-K value.

### `start-server`

The local `start-server` sets defaults so env variables do not need to be
exported manually every time:

- hot/cold parallelization uses the code default `auto`
- `LLAMA_MOE_HOT_CACHE_PARALLEL_MIN_SLOTS=0`
- `LLAMA_MOE_HOT_CACHE_JSON=/home/adrian/models/heatmap-data.json`
- `LLAMA_MOE_HOT_CACHE_MAX_MIB=8000`

It then starts `llama-server` with:

- `--models-preset /home/adrian/llama.exp/model_config.ini`
- `--moe-hot-cache-max-mib "$LLAMA_MOE_HOT_CACHE_MAX_MIB"`
- `--moe-hot-cache "$LLAMA_MOE_HOT_CACHE_JSON"`

### `start-server-performance14`

A second local variant starts with:

- `LLAMA_MOE_HOT_CACHE_JSON=/home/adrian/llama.exp/performance14.json`

That was useful for reproducible testing against a specific hot-cache JSON.

## Tests

`tests/test-moe-hot-cache.cpp` was rebuilt or extended. It covers:

- parsing and sorting old perf JSON data,
- parsing hot/cold branch counts plus layer weighting,
- expert budget selection,
- errors for invalid schema,
- mixed hot/cold worklist,
- all-hot and all-cold cases,
- worklist creation directly from router logits including weighting.

Why: the hot/cold path is very sensitive to small layout mistakes. If a count,
slot, or padding value is wrong, the result is either wrong output or scheduler
regions that cannot be parallelized.

Impact: the tests protect the logic that can sensibly be tested without a large
Qwen model. The truly backend-specific parts still need real server/CLI runs.

## Performance Progression

Rough progression from the user-reported runs:

| Phase | Result | Meaning |
| --- | ---: | --- |
| Without parallelization, simple Hallo | about `19.7 tk/s` | comparison value for normal path |
| First parallel variant, simple Hallo | about `13.96 tk/s` | correct, but too much overhead |
| First programming runs | about `16.4-17 tk/s` | hot cache fits programming prompt better |
| `performance9` / `performance14` | about `20 tk/s` | first large parallel gains |
| `performance41-47` | about `23.7-24.75 tk/s` | scheduler/routing/cache selection improved |
| `performance50-52` | about `25.45-25.49 tk/s` | further reduced cold/merge work |
| `performance53` | almost `26 tk/s` | weighted prefix sum and cold-path work |
| `performance54` | about `27.1 tk/s` | graph pruning and fewer unused views |
| `performance55` | about `27.76 tk/s` | repeat-hot-input |
| `performance56` | about `28.09 tk/s` | current state including cold-first-row test |

## Current Bottlenecks According To `performance56.json`

The most important summary values:

| Metric | Value |
| --- | ---: |
| `total_moe_time_per_call_us` | `816.307` |
| `routing_time_per_call_us` | `264.614` |
| `worklist_time_per_call_us` | `65.3677` |
| `merge_time_per_call_us` | `471.037` |
| `hot_gather_scatter_time_per_call_us` | `5.00415` |
| `cold_gather_scatter_time_per_call_us` | `2.56514` |
| `parallel_region_wall_time_per_call_us` | `422.892` |
| `parallel_hot_lane_wall_time_per_call_us` | `123.504` |
| `parallel_cold_lane_wall_time_per_call_us` | `382.102` |
| `parallel_join_wait_time_per_call_us` | `332.197` |
| `parallel_fallbacks` | `0` |

Interpretation:

- Merge is the largest reported individual block at about 58% of
  `total_moe_time_per_call_us`.
- Routing is still large at about 32%.
- Worklist became much smaller, but is still visible.
- Gather/scatter is small after the last optimizations.
- The parallel region works without fallbacks.
- The cold lane is much longer than the hot lane; the join therefore still
  waits heavily on CPU cold work.

## What The Work Achieved Overall

Functionally:

- The hot cache can be planned and loaded from perf JSON.
- Qwen3.5/Qwen3.6-MoE can compute hot and cold experts separately.
- CPU cold and CUDA hot run in parallel in a scheduler region.
- First-inference crashes and split-order errors were fixed.
- `--no-perf` can disable performance counters for pure speed runs.

Performance:

- The first working variant was still slower than the normal path.
- Through scheduler, routing, worklist, merge, and cold-path work, the
  programming prompt increased from about `16-17 tk/s` to about `28.09 tk/s`.
- Compared with the early working parallel path, this is roughly `+65%`.
- Compared with the simple `19.7 tk/s` reference value, the current programming
  run is roughly `+43%`, although this is not a clean A/B comparison.

Diagnostics:

- Perf data now shows that the heatmap itself is no longer the problem; merge,
  cold lane, join wait, and routing are.
- The hot hit rate is close to the expected limit; further large gains probably
  need less overhead and a shorter cold tail.

## Next Technical Levers

The most sensible next directions from today's perspective are:

1. Reduce merge further
   - `merge_time_per_call_us` is the largest block.
   - Possible direction: fold hot and cold results earlier into a shared
     `[n_embd, 1]` result, with fewer add/sum/view nodes.

2. Shorten the cold lane
   - The cold lane determines the join.
   - Possible direction: combine CPU-cold matmul and prefix sum more tightly,
     fewer intermediate layouts, better active-ID list.

3. Reduce routing
   - Routing is still around `265 us` per layer call.
   - Possible direction: specialize Top-K for `n_tokens == 1` further, fewer
     generic tensor paths, maybe direct router evaluation in an even narrower op.

4. Clarify Cold-First-Row A/B
   - Net result in `performance56.json` is slightly positive, but with higher
     merge and join wait.
   - Should remain as a toggle and be tested with multiple comparable runs.

5. Dynamic cache
   - Updating after inference runs could improve selection, but it will only
     help if it hits cold-tail layers.
   - Since hot hit rate is probably capped near 70%, dynamic update should not
     optimize for maximum hit rate but for reduced join wait.
