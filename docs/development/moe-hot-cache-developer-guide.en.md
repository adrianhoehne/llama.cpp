# MoE Hot Cache: Developer Guide

Status: 2026-05-15  
Branch: `cached-experts-v2`  
Last referenced commit: `e3ace0d3b Separate moe hot cache feature pt5.`

This document is the English developer guide for the experimental MoE hot-cache work in this fork. It explains what was changed, why it was changed, how the runtime path works, which switches matter, how to interpret the performance data, and where the remaining maintenance risks are.

The implementation targets Qwen3.5/Qwen3.6 MoE decode throughput. The core idea is to cache frequently used experts on the GPU and run the remaining cold expert work on the CPU in parallel.

## Summary

The change adds an optional hot cache for MoE experts.

Without the hot cache, the server should behave like the previous llama.cpp path. The experimental path only becomes active when a non-zero hot-cache budget is configured:

```text
--moe-hot-cache-max-mib > 0
```

When a budget is set, a hot-cache JSON file is required as well:

```text
--moe-hot-cache <file.json>
```

The hot-cache path:

1. reads a performance JSON with layer and expert usage data,
2. selects hot experts per layer within a memory budget,
3. copies the selected experts into dedicated hot-cache tensors,
4. builds a specialized Qwen MoE graph for active layers,
5. splits runtime MoE work into hot and cold expert slots,
6. runs hot work on the GPU and cold work on the CPU in parallel,
7. merges both branches back into the normal tensor flow.

The main goals were:

- improve GPU utilization during decode,
- avoid running all cold expert work serially after GPU work,
- reduce decode overhead,
- keep performance counters focused on optimization data,
- isolate experimental code from upstream-sensitive llama.cpp core files.

## Current Performance State

These numbers are development observations, not formal benchmarks. They are still useful because they show the direction of the optimization work.

| State | Observation |
| --- | --- |
| No hot/cold parallelization, simple "Hallo" prompt | about 19.7 tok/s |
| First parallel proof of concept with an unfavorable hot list | about 13.96 tok/s on a simple prompt |
| Early programming-task runs | about 16 to 17 tok/s |
| After scheduler, routing, and merge improvements | above 20 tok/s |
| Later `performance40` to `performance52` runs | about 23.7 to 25.5 tok/s |
| `performance53` | almost 26 tok/s |
| `performance54` | about 27.1 tok/s |
| `performance55` | about 27.76 tok/s |
| `performance56` | about 28.09 tok/s |

The main gain did not come from one single patch. It came from the combination of:

- lower decode routing overhead,
- lower merge overhead,
- fewer unnecessary branch launches,
- a faster cold path,
- explicit scheduler regions for hot/cold work,
- reduced performance counter overhead,
- strict gating so the normal path remains intact.

## Terminology

`hot experts`  
Experts that are frequent or expensive enough in the performance data to be copied into the GPU hot cache.

`cold experts`  
Experts that were not selected or did not fit into the memory budget. They stay in the normal model layout and are preferably processed by the CPU path in the parallel mode.

`hot slot ratio`  
The fraction of MoE slots served by hot experts. In the current data, the practical expectation was roughly 68 to 70 percent. That means the remaining overhead must be low: the GPU hit rate will not reach 100 percent with this static cache strategy.

`slot`  
A concrete token-expert assignment. With `n_expert_used = 8`, each token can create up to eight expert slots. Increasing `n_expert_used` to 16 roughly doubles this slot work, assuming the model and architecture allow it.

`decode`  
The one-token path during token generation. This is where overhead matters most.

`prefill`  
The prompt-processing path with many tokens. Its cost profile is different from decode.

## Activation And Gating

The hot cache is explicitly gated. This is critical because the fork must still support the normal llama.cpp behavior when the experiment is not enabled.

The new path is disabled when:

```text
moe_hot_cache_max_mib == 0
```

In that case:

- `llama_moe_hot_cache_init(...)` returns without building a cache,
- `model.moe_hot_cache` stays empty,
- Qwen35Moe uses the normal `build_moe_ffn` path,
- the hot-cache graph is not built,
- the scheduler does not see a hot-cache parallel region.

The new path is active when both are provided:

```text
--moe-hot-cache-max-mib <N>
--moe-hot-cache <file.json>
```

If a budget is set but no JSON file is provided, that is intentionally treated as an error. Otherwise the runtime would not know which experts should be cached.

The Qwen hot path is built per layer only when:

```text
llama_moe_hot_cache_layer_active(model, il)
```

returns true.

This allows some layers to use the hot path while other layers still use the normal path.

## Runtime Switches

### Hot Cache Selection

```text
--moe-hot-cache <file.json>
LLAMA_ARG_MOE_HOT_CACHE=<file.json>
```

Path to the performance or hot-cache JSON.

```text
--moe-hot-cache-max-mib <N>
LLAMA_ARG_MOE_HOT_CACHE_MAX_MIB=<N>
```

Maximum memory budget for the hot cache in MiB. `0` disables the feature.

### Parallelization

```text
LLAMA_MOE_HOT_CACHE_PARALLEL=1
```

Enables hot/cold parallelization in auto mode.

```text
LLAMA_MOE_HOT_CACHE_PARALLEL=force
```

Enables force mode. If the region cannot be parallelized correctly, the scheduler reports an error instead of silently falling back to serial execution. This is useful for debugging but less suitable for normal performance runs.

```text
LLAMA_MOE_HOT_CACHE_PARALLEL=0
```

Disables hot/cold parallelization.

```text
LLAMA_MOE_HOT_CACHE_PARALLEL_MIN_SLOTS=<N>
```

Minimum number of slots required before the scheduler launches the hot/cold parallel region. Default: `64`.

This avoids paying thread and scheduler overhead for tiny work packets.

### Decode And Merge Optimizations

The following switches exist as development levers. Many are currently enabled by default because they improved the later performance runs.

```text
LLAMA_MOE_HOT_CACHE_CPU_DECODE_ROUTING=1
```

Uses a CPU custom-op path for decode routing.

```text
LLAMA_MOE_HOT_CACHE_DECODE_DIRECT_MERGE=1
```

Enables the direct decode merge path.

```text
LLAMA_MOE_HOT_CACHE_DECODE_STRIDED_SUM_ROWS=1
```

Enables an optimized decode merge sum path.

```text
LLAMA_MOE_HOT_CACHE_HOT_DUMMY_PADDING=1
```

Adds hot-path padding so graph and backend expectations stay stable.

```text
LLAMA_MOE_HOT_CACHE_SHARED_INPUT_ROW=1
```

Enables the shared-input-row optimization for the hot path.

```text
LLAMA_MOE_HOT_CACHE_COLD_PREFIX_SUM=1
LLAMA_MOE_HOT_CACHE_COLD_PREFIX_WEIGHTED_SUM=1
```

Enables cold-path prefix-sum optimizations.

```text
LLAMA_MOE_HOT_CACHE_DECODE_REPEAT_HOT_INPUT=1
LLAMA_MOE_HOT_CACHE_COLD_FIRST_ROW_INPUT=1
```

Enables additional decode-specific reductions of gather and input overhead.

### Performance Counters

```text
--no-perf
```

Disables llama.cpp performance counters and the MoE performance collection path for a cleaner throughput run.

```text
LLAMA_MOE_LAYER_PERF=0
```

Disables the MoE layer performance output.

```text
LLAMA_MOE_LAYER_PERF_EXPERT_COUNTS=1
```

Enables detailed expert counts. This is useful when creating the first hot-cache JSON from real traffic, but it should be off for speed tests.

## File Map

### `src/llama-moe-hot-cache.h`

Defines the central hot-cache data structures:

- cache configuration,
- layer configuration,
- expert selection data,
- tensor and worklist layouts,
- helpers for checking whether a layer is active.

Important worklist fields:

```text
HOT_ID
HOT_SRC_SLOT
HOT_TOKEN_ID
HOT_WEIGHT
COLD_ID
COLD_SRC_SLOT
COLD_TOKEN_ID
COLD_WEIGHT
HOT_EXPERT_ID
HOT_COUNT
COLD_COUNT
```

The worklist separates hot and cold slots. This lets the scheduler and merge logic handle both branches independently.

### `src/llama-moe-hot-cache.cpp`

Responsible for:

- parsing the JSON file,
- recognizing supported schemas,
- collecting expert sizes,
- scoring experts,
- selecting experts within the memory budget,
- allocating hot-cache tensors,
- copying selected experts,
- building mapping and mask tables.

Supported schemas:

```text
llama.cpp.moe_layer_perf.v1
llama.cpp.moe_layer_opt_perf.v1
```

The selection is not just "most frequently used experts". Later versions also account for layer wait time and a sticky bonus for already selected hot experts. That made the cache set more stable and more representative of actual bottlenecks.

### `src/llama-moe-hot-cache-graph.cpp`

Contains the Qwen35Moe-specific hot-cache graph implementation.

This was an important refactor. The experimental graph logic was moved out of `src/models/qwen35moe.cpp`, so the model file stays closer to upstream.

Responsibilities:

- build the hot/cold FFN for Qwen35Moe,
- build decode-specific worklists,
- wire CPU custom ops for routing and merge,
- build the hot branch using cached experts,
- build the cold branch using normal experts,
- merge both branch outputs,
- annotate the scheduler parallel region,
- set local `mul_mat_id` flags.

This code is intentionally Qwen35Moe-specific. It is not a generic MoE replacement for all architectures.

### `src/models/qwen35moe.cpp`

This file is small again after the refactor.

The central decision is:

```text
if the layer has an active hot cache:
    build_layer_ffn_hot(...)
else:
    normal build_moe_ffn(...)
```

The normal model path remains available.

### `src/llama-moe-hot-cache-perf.h`

Declares the MoE performance interfaces.

The important public function is:

```text
llama_moe_layer_perf_json(ctx)
```

It is also exposed through the public API in `include/llama.h`.

### `src/llama-moe-hot-cache-perf.cpp`

Contains MoE-specific performance collection and JSON output.

Before the refactor, relevant logic lived in `llama-context.cpp`. Moving it here keeps the core context code closer to upstream.

Collected data includes:

- global summary,
- layer calls,
- hot slot ratio,
- routing time,
- worklist time,
- merge time,
- hot branch time,
- cold branch time,
- hot and cold matmul time,
- gather and scatter times,
- parallel-region times,
- join wait time,
- overlap,
- launch and fallback counts,
- optional expert counts.

When performance collection is disabled, the JSON function intentionally returns only a small disabled block:

```json
{
  "enabled": false,
  "schema": "llama.cpp.moe_layer_opt_perf.v1",
  "layers": []
}
```

### `src/llama-context.cpp`

Only small hooks remain:

- set the performance callback,
- enable scheduler performance collection,
- compute the graph as before.

When `--no-perf` is active, the MoE performance path is not installed.

### `src/llama.cpp`

Initializes the hot cache when the model is loaded:

```text
llama_moe_hot_cache_init(*model, params)
```

The function is gated by `moe_hot_cache_max_mib`.

### `src/llama-model.cpp` And Related Parameter Files

Extend model parameters with:

```text
moe_hot_cache_path
moe_hot_cache_max_mib
```

The model owns the hot-cache state:

```text
std::unique_ptr<llama_moe_hot_cache> moe_hot_cache
```

### `src/llama-graph.cpp` And `src/llama-graph.h`

These files were moved closer to upstream again.

Earlier hot-cache-specific extensions were removed from the generic graph layer, including:

- a special `build_moe_ffn_with_ids`,
- `llm_mul_mat_id_flags`,
- hot-cache-specific `build_lora_mm_id` behavior.

This reduces conflicts during upstream updates.

### `ggml/include/ggml-backend-moe-hot-cache.h`

New experimental header for the hot-cache scheduler API.

The reason for a separate header is to keep `ggml-backend.h` from accumulating experimental API surface.

### `ggml/src/ggml-backend-moe-hot-cache.inc`

Private implementation of the scheduler extension.

This `.inc` file is included from `ggml-backend.cpp`. That keeps access to internal scheduler structures while moving the large experimental implementation out of the core backend file.

Responsibilities:

- register parallel regions,
- force split boundaries,
- validate regions,
- identify hot and cold splits,
- run a CPU worker for the cold path,
- let the normal thread/GPU backend run the hot path,
- join both paths,
- collect errors and fallback reasons,
- collect scheduler performance data.

### `ggml/src/ggml-backend.cpp`

Only small integration points remain:

- scheduler state has a hot-cache state pointer,
- init/free/reset hook that state,
- split-boundary logic calls the hot-cache extension,
- the `.inc` implementation is included.

This is intentionally small to reduce upstream merge conflicts.

### `ggml/src/ggml-cpu/ggml-cpu.c`

CPU backend changes for hot-cache MoE work.

Relevant points:

- `MUL_MAT_ID` can be controlled through `op_params` for hot-cache layouts,
- negative IDs can be allowed,
- zeroing can be skipped when a path is known not to write output,
- certain decode layouts with shared input rows are supported.

This is one of the places that cannot be completely isolated outside the core.

### `ggml/src/ggml-cuda/ggml-cuda.cu`

CUDA backend changes for `MUL_MAT_ID` variants needed by the hot path.

Relevant points:

- hot-cache flags from `op_params`,
- support for special ID layouts,
- avoidance of unnecessary work in known decode cases.

This is also a core hook and therefore a possible conflict area during upstream updates.

### `tests/test-moe-hot-cache.cpp`

Unit tests for:

- JSON selection,
- budget and layer behavior,
- basic worklist and mapping behavior,
- gating cases.

Future decode-path optimizations should extend this test file when possible.

## Runtime Flow

### 1. Startup Parameters Are Parsed

The CLI/common parameter layer reads the hot-cache arguments and performance switches.

Important:

```text
--moe-hot-cache-max-mib 0
```

is the default and means: no hot cache.

### 2. The Model Is Loaded

When `moe_hot_cache_max_mib > 0`, the hot cache is initialized during model loading.

The cache needs:

- model metadata,
- layer and expert sizes,
- performance JSON data,
- memory budget,
- backend information.

The current experimental parallel path expects hot work on CUDA and cold work on CPU.

### 3. Hot Experts Are Selected

The JSON file contains layer and expert data. The cache uses it to build a ranked selection.

The goal is not simply selecting the most frequent experts. The selection considers:

- how often an expert is used,
- which layer it belongs to,
- whether the layer causes wait time,
- how expensive the expert is in memory,
- whether it fits a stable hot set.

The memory budget limits the final set.

### 4. Hot-Cache Tensors Are Built

Selected experts are copied into hot-cache tensors.

This lets the hot path use a denser layout and avoid going through the normal expert layout for those experts.

### 5. The Graph Is Built

For every Qwen35Moe layer:

```text
if hot cache is active for this layer:
    build_layer_ffn_hot(...)
else:
    build_layer_ffn(...)
```

The normal path stays available per layer.

### 6. The Worklist Is Computed

In decode, `n_tokens == 1`. This path was optimized heavily.

The worklist separates:

- hot slots,
- cold slots,
- weights,
- source slot,
- token ID,
- expert ID.

This separation is the foundation for parallel execution.

### 7. Hot And Cold Branches Are Built

Hot branch:

- uses cached experts,
- is intended for CUDA/GPU,
- should cause as little CPU overhead as possible.

Cold branch:

- uses normal experts,
- is intended for CPU,
- runs in parallel with GPU work when enough slots are available.

### 8. The Scheduler Region Is Annotated

The graph marks the area as a hot-cache parallel region:

```text
ggml_backend_sched_moe_hot_cache_parallel_region(...)
```

The annotation includes:

- layer ID,
- hot-count tensor,
- cold-count tensor,
- start and end tensors,
- output and join tensors.

### 9. The Scheduler Validates The Region

The scheduler parallelizes only if the region is recognized cleanly.

Validation checks:

- the region is complete,
- count tensors are in the expected prefix,
- split order is valid,
- hot and cold regions are separate,
- hot is on CUDA,
- cold is on CPU,
- backends are not identical,
- count readback works,
- the minimum slot threshold is reached.

In auto mode, the scheduler can fall back to serial execution. In force mode, the error is visible.

An early error looked like this:

```text
region split order is not hot-then-cold-then-join
```

At that point, the scheduler expected a stricter split order than the graph produced. Later refactors made the recognition and validation more robust.

### 10. Parallel Compute Runs

The threads are CPU threads.

Important: the GPU does not run CPU threads. The flow is:

- the main thread starts or manages hot work on the CUDA backend,
- a CPU worker processes cold work,
- CUDA work executes asynchronously on the GPU,
- CPU and GPU work overlap in time,
- the join synchronizes both branches.

Starting or waking the worker has overhead. This is why `LLAMA_MOE_HOT_CACHE_PARALLEL_MIN_SLOTS` exists.

### 11. Results Are Merged

After hot and cold work, both outputs are merged into the normal FFN output.

The merge was a major optimization lever because it runs once per decode step and per layer. The later direct decode-merge paths reduce unnecessary tensor work.

## Scheduler Fallbacks

The performance data can contain fallback reasons.

Typical reasons:

| Reason | Meaning |
| --- | --- |
| `incomplete` | The region was not fully annotated |
| `count_not_prefix` | Count tensors were not in the expected prefix |
| `bad_split_order` | Splits were not in a valid hot/cold/join order |
| `same_backend` | Hot and cold ended up on the same backend |
| `hot_spans_backends` | Hot work was not assigned to one clear backend |
| `cold_spans_backends` | Cold work was not assigned to one clear backend |
| `hot_not_cuda` | Hot work was not on CUDA |
| `cold_not_cpu` | Cold work was not on CPU |
| `count_readback` | Count readback failed |
| `threshold` | Too few slots for parallelization |
| `zero_output` | Output had to be zeroed because a branch was empty |

A good run should have few or no unexpected fallbacks. Threshold fallbacks can be normal when work packets are too small to justify parallel execution.

## Performance JSON

The performance output was intentionally reduced. Earlier data was more useful for visualizing layer and expert heatmaps. For optimization, we need different data:

- where time is spent,
- how much hot and cold work exists,
- how well CPU and GPU overlap,
- how expensive routing and merge are,
- whether scheduler fallbacks happen,
- whether the hot list actually hits.

Current schema:

```text
llama.cpp.moe_layer_opt_perf.v1
```

Important fields:

```text
summary.layer_calls
summary.hot_slot_ratio
summary.total_moe_us_per_call
summary.routing_us_per_call
summary.worklist_us_per_call
summary.merge_us_per_call
summary.parallel_region_us_per_call
summary.parallel_hot_us_per_call
summary.parallel_cold_us_per_call
summary.parallel_join_wait_us_per_call
summary.parallel_overlap_us_per_call
summary.parallel_launches
summary.parallel_fallbacks
layers[]
```

With:

```text
LLAMA_MOE_LAYER_PERF_EXPERT_COUNTS=1
```

detailed expert counters are included. They are useful for initial cache generation but too expensive for clean speed runs.

## Creating The First Hot-Cache JSON

If no hot-cache JSON exists yet, start without the hot cache but with expert counts enabled:

```bash
LLAMA_MOE_LAYER_PERF_EXPERT_COUNTS=1 \
./build/bin/llama-server \
  --perf \
  <normal model and server arguments>
```

Then run representative prompts and read the MoE performance JSON from the server, for example through the existing `/moe-layer-perf` path.

That JSON can then be used as input for the hot cache:

```bash
LLAMA_MOE_HOT_CACHE_PARALLEL=1 \
./build/bin/llama-server \
  --moe-hot-cache performance.json \
  --moe-hot-cache-max-mib <budget> \
  <normal model and server arguments>
```

For final throughput measurements:

```bash
LLAMA_MOE_HOT_CACHE_PARALLEL=1 \
./build/bin/llama-server \
  --no-perf \
  --moe-hot-cache performance.json \
  --moe-hot-cache-max-mib <budget> \
  <normal model and server arguments>
```

## Why `--no-perf` Matters

Performance counters are not free.

Even when the JSON endpoint is queried rarely, decode still has to update measurement points, counters, and sometimes scheduler data. For very small decode steps, this overhead is visible.

There are therefore two useful modes.

Development mode:

```text
--perf
LLAMA_MOE_LAYER_PERF=1
```

Goal: understand where time is spent.

Speed mode:

```text
--no-perf
LLAMA_MOE_LAYER_PERF=0
```

Goal: measure real throughput without measurement overhead.

## Why The First Parallel Proof Of Concept Was Slower

The first working parallel path was intentionally not optimal. It proved that the split could work, but it still had a lot of overhead:

- routing was too expensive,
- merge was too expensive,
- scheduler start and join were too expensive,
- very small work packets were parallelized,
- the hot list did not fit every prompt,
- detailed performance counters were enabled,
- the cold path had not yet been optimized.

This made a simple prompt slower even while programming tasks already showed more promising behavior. That was expected because the hot list came from programming traffic.

## Why Later Runs Became Faster

The later improvements came from several levers.

### 1. Reduced Decode Routing

The decode path has only one token. General MoE graph logic for many tokens is too expensive here.

A specialized decode worklist path was added. It reduces:

- unnecessary tensor operations,
- unnecessary copies,
- unnecessary sorting and mapping work.

### 2. Reduced Merge Cost

Merge runs after every hot/cold branch. If it is inefficient, it consumes the benefit of parallelization.

Optimizations included:

- direct decode merge,
- strided sum rows,
- less zeroing,
- better use of known decode layouts.

### 3. Parallel Regions Only For Enough Work

The CPU worker has to be started or woken. This is not worth it for only a few slots.

`LLAMA_MOE_HOT_CACHE_PARALLEL_MIN_SLOTS` avoids small and expensive parallel regions.

### 4. Faster Cold Path

Because the hot hit rate cannot reach 100 percent, the cold path remains important.

Cold-path improvements were especially valuable because the join waits for CPU work when the cold path is too slow.

### 5. Minimal Performance Path

The performance output was changed from visualization-oriented data to optimization-oriented data.

It remains useful in development mode but causes less overhead. For final measurements it can be disabled completely.

### 6. Refactor For Isolation

Experimental code was moved into dedicated files. This does not directly improve tok/s, but it makes future optimization work safer and easier to maintain.

## Refactor History

The following commits describe the rough timeline since the likely starting point `0641adb9e08da0a675058bc39a8c928a7f8d6ad0`.

| Commit | Purpose |
| --- | --- |
| `5ad16adeb` | First inference-test changes, functional but slower |
| `c26ec8556` | Runnable proof of concept |
| `61aaa4c75` | Added more metrics |
| `338c5ee05` | Broken parallelization attempt |
| `2a795fc3a` | Reverted the broken attempt |
| `dac2548ac` | Computed CPU and GPU work separately in parallel |
| `39a942ab1` | Scheduler optimization |
| `50284e9b8` | Speed started moving in the right direction |
| `220b44da3` | Different method for selecting hot experts |
| `5759b13aa` | Adjusted weights |
| `03b23a2f6` | Improved routing |
| `9987d8266` | Reduced performance path to the minimum |
| `19c309c84` | Made performance collection disableable |
| `d7305c2d0` | More performance improvements |
| `a47d68214` | More performance for the cold path |
| `4b47c82c5` | New state around 28 tok/s |
| `e91fb763a` | Refactor part 1: separated Qwen hot graph |
| `3de19d8ae` | Refactor part 2: made generic graph code closer to upstream |
| `106e407ea` | Refactor part 3: separated performance code |
| `0c3b4a58e` | Refactor part 4: separated scheduler implementation |
| `e3ace0d3b` | Refactor part 5: separated experimental scheduler API |

## Refactor Details

### Part 1: Qwen Hot Graph Was Separated

Before this refactor, much of the hot-cache graph logic lived directly in `src/models/qwen35moe.cpp`.

Problems:

- the model file became large,
- upstream conflicts were likely,
- it was hard to see which code was experimental.

Solution:

- created `src/llama-moe-hot-cache-graph.cpp`,
- reduced `qwen35moe.cpp` to a small gate,
- kept the normal Qwen path visible and unchanged.

### Part 2: Generic Graph Code Was Moved Closer To Upstream

Before this refactor, hot-cache-specific flags and helpers lived in `llama-graph.*`.

Problems:

- generic graph code mixed with experimental semantics,
- other models appeared indirectly affected,
- upstream merges became harder.

Solution:

- removed or moved hot-cache-specific `build_moe_ffn_with_ids`,
- kept `build_lora_mm_id` generic again,
- managed `mul_mat_id` flags locally in the hot-cache graph.

### Part 3: Performance Code Was Separated

Before this refactor, MoE performance logic lived in the context code.

Problems:

- `llama-context.cpp` became harder to read,
- performance collection was harder to disable cleanly,
- upstream conflict surface was too large.

Solution:

- added `src/llama-moe-hot-cache-perf.h`,
- added `src/llama-moe-hot-cache-perf.cpp`,
- left only small hooks in the context.

### Part 4: Scheduler Implementation Was Separated

Before this refactor, much of the hot-cache scheduler logic lived in `ggml-backend.cpp`.

Problems:

- `ggml-backend.cpp` is upstream-sensitive,
- large local changes create merge conflicts,
- the experimental scheduler logic was hard to identify.

Solution:

- added `ggml/src/ggml-backend-moe-hot-cache.inc`,
- kept a small state pointer in the scheduler,
- kept small init/free/reset/split hooks in the core file.

### Part 5: Experimental Scheduler API Was Separated

Before this refactor, the hot-cache API lived in `ggml-backend.h`.

Problems:

- the public core header gained experimental API,
- staying close to upstream was harder.

Solution:

- added `ggml/include/ggml-backend-moe-hot-cache.h`,
- hot-cache code includes this header explicitly,
- the general backend header stays cleaner.

## Core Hooks That Still Remain

Full isolation is not possible as long as we want real parallel execution inside the existing graph and scheduler.

The following hooks intentionally remain in core files:

- scheduler state in `ggml-backend.cpp`,
- split-boundary detection,
- inclusion of the private scheduler extension,
- CPU and CUDA `MUL_MAT_ID` semantics for hot-cache layouts,
- model parameters for hot-cache CLI support,
- small context hooks for performance collection.

These hooks should remain small. New hot-cache logic should preferably go into:

```text
src/llama-moe-hot-cache*.*
ggml/src/ggml-backend-moe-hot-cache.inc
ggml/include/ggml-backend-moe-hot-cache.h
```

## Upstream Update Strategy

When rebasing or merging upstream llama.cpp:

1. keep conflicts in generic files as small as possible,
2. check whether scheduler structures in `ggml-backend.cpp` changed,
3. check whether `ggml-cpu.c` or `ggml-cuda.cu` changed `MUL_MAT_ID`,
4. compare the Qwen35Moe model file around the small hot-cache gate,
5. test the hot-cache-specific files afterwards.

Most conflict-prone files:

```text
ggml/src/ggml-backend.cpp
ggml/src/ggml-cpu/ggml-cpu.c
ggml/src/ggml-cuda/ggml-cuda.cu
src/llama.cpp
src/llama-context.cpp
src/llama-model.cpp
src/models/qwen35moe.cpp
```

Less conflict-prone files:

```text
src/llama-moe-hot-cache.cpp
src/llama-moe-hot-cache.h
src/llama-moe-hot-cache-graph.cpp
src/llama-moe-hot-cache-perf.cpp
src/llama-moe-hot-cache-perf.h
ggml/src/ggml-backend-moe-hot-cache.inc
ggml/include/ggml-backend-moe-hot-cache.h
tests/test-moe-hot-cache.cpp
```

## Development Workflow

### Build

Preferred build command in this repository:

```bash
cmake --build build -j8
```

For faster checks of individual targets:

```bash
cmake --build build -j8 --target llama
cmake --build build -j8 --target test-moe-hot-cache
```

### Unit Test

```bash
./build/bin/test-moe-hot-cache
```

### Server Without Hot Cache

This mode should behave like normal llama.cpp:

```bash
./build/bin/llama-server <normal arguments>
```

Important: do not set `--moe-hot-cache-max-mib`.

### Server With Hot Cache And Performance Counters

```bash
LLAMA_MOE_HOT_CACHE_PARALLEL=1 \
./build/bin/llama-server \
  --perf \
  --moe-hot-cache performance.json \
  --moe-hot-cache-max-mib <budget> \
  <normal arguments>
```

### Server With Hot Cache And No Performance Counters

```bash
LLAMA_MOE_HOT_CACHE_PARALLEL=1 \
./build/bin/llama-server \
  --no-perf \
  --moe-hot-cache performance.json \
  --moe-hot-cache-max-mib <budget> \
  <normal arguments>
```

### Optional CLI Test

The CLI test was useful for quick reproduction, but it was not always stable in this environment.

```bash
/home/adrian/llama.exp/build/bin/llama-cli \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q6_K_XL \
  -dev CUDA0 \
  --ctx-size 32000 \
  -p "Antworte mit Hallo" \
  -st
```

The server is currently the more reliable test path.

## Reading Performance Data

When inspecting a new `performance*.json`, the most important questions are:

1. What is `hot_slot_ratio`?
2. Are there unexpected scheduler fallbacks?
3. Is `parallel_join_wait_us_per_call` high?
4. Is the cold path too slow?
5. Is merge cost visible again?
6. Is routing or worklist time visible in decode?
7. Do hot and cold branches actually overlap?
8. Does `--no-perf` change throughput significantly?

If `hot_slot_ratio` is effectively capped around 70 percent, the next large gains must come from lower overhead and a faster cold path. More hot cache alone will have limited value.

## What Happens With More `n_expert_used`

If `n_expert_used` increases from 8 to 16, each token creates more expert slots.

Expected effects:

- more total work,
- more routing and worklist work,
- more merge work,
- potentially more useful parallel work,
- also more cold work if the hot cache does not hit well,
- the hot-cache memory budget becomes tight sooner.

More experts per token is not an automatic speed improvement. It may change model quality or behavior, but this optimization path must be measured again with the new slot distribution.

## Known Limits

### Static Cache

The hot cache is currently static. It is built once at startup from a JSON file.

Dynamic updates after each inference run are not implemented.

A dynamic cache is possible but significantly more complex:

- new expert selection during runtime,
- safe tensor reallocation or double-buffered cache,
- synchronization with currently running graphs,
- background GPU/CPU copies,
- protection against unstable hot sets.

### CUDA/CPU Assumption

The parallel path is designed for:

```text
hot: CUDA
cold: CPU
```

Other backends are not the goal of this code. Vulkan, Metal, or multi-CUDA-GPU support would require changes to scheduler validation and graph assignment.

### Qwen35Moe Specificity

The hot graph is built for Qwen35Moe. Other MoE models use the normal path unless they are explicitly wired in.

### Workload-Dependent Performance Data

A hot list created from programming tasks can be worse for simple prompts. Hot-cache JSON files should be created from representative traffic.

## Typical Failure Modes

### Assertion In `ggml_view_1d`

Earlier failure:

```text
GGML_ASSERT(view_src == NULL || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src)) failed
```

The cause was an invalid view/worklist layout in the first inference path. The fix was to size and split hot/cold worklist regions correctly.

### Scheduler Reports Bad Split Order

Failure:

```text
forced MoE hot-cache parallel region failed for layer 0:
region split order is not hot-then-cold-then-join
```

The scheduler expected a stricter region order than the graph produced. Later scheduler recognition and graph annotation made this more robust.

### Server Hangs During Warmup

Possible causes:

- force mode with a region that cannot validate,
- overly aggressive parallel region,
- count readback or synchronization issue,
- warmup uses a different slot distribution than decode,
- performance or expert counters create unexpected load.

Debug order:

1. test without `LLAMA_MOE_HOT_CACHE_PARALLEL`,
2. use auto mode instead of `force`,
3. increase `LLAMA_MOE_HOT_CACHE_PARALLEL_MIN_SLOTS`,
4. test `--no-perf`,
5. inspect performance JSON fallback counts.

## Rules For Further Optimization

### 1. Measure First

Every new optimization should compare at least:

- final tok/s after a longer run,
- `hot_slot_ratio`,
- `parallel_join_wait_us_per_call`,
- `parallel_overlap_us_per_call`,
- `cold_branch_us_per_call`,
- `merge_us_per_call`,
- `routing_us_per_call`,
- fallback counts.

### 2. Avoid Touching Normal Paths When Possible

If a change only affects hot cache behavior, it should live in hot-cache files.

### 3. Keep Core Hooks Small

Changes in `ggml-backend.cpp`, `ggml-cpu.c`, or `ggml-cuda.cu` should be minimal and well justified.

### 4. Decode Matters More Than Prefill

The main goal is token generation/decode. Prefill must not break, but the largest benefit is in the one-token path.

### 5. Force Mode Is For Debugging

`LLAMA_MOE_HOT_CACHE_PARALLEL=force` is useful for surfacing validation errors immediately. Auto mode is safer for normal performance runs.

## Remaining Optimization Levers

### Further Reduce The Cold Path

If hot hit rate is capped around 70 percent, about 30 percent of the work remains cold. Any cold-path reduction directly reduces join wait.

Possible directions:

- less gather work in the cold path,
- better batch or slot compaction,
- less separate weight application,
- more decode specialization.

### Reduce Join Wait

If the GPU finishes early and waits for CPU work, cold work must become smaller or start earlier.

Possible directions:

- move count/worklist creation so cold work can start earlier,
- reduce the cold branch,
- further amortize scheduler start costs.

### Improve Hot-Cache Selection

With more data, selection can improve.

Possible directions:

- weight layer wait time more strongly,
- separate prompt classes,
- maintain multiple cache profiles,
- stabilize hot sets so single outliers do not dominate.

### Dynamic Cache

Long term, a dynamic cache could help when workloads change significantly. Short term, it is more complex than continuing to reduce overhead.

### Backend-Specific `MUL_MAT_ID` Paths

Some overhead remains in `MUL_MAT_ID` special cases. There may still be performance here, but this area is core-adjacent and conflict-prone.

## Minimum Checklist For New Changes

Before committing:

```bash
cmake --build build -j8 --target llama
cmake --build build -j8 --target test-moe-hot-cache
./build/bin/test-moe-hot-cache
```

If runtime behavior changed:

1. start the server without hot cache and test a simple inference,
2. start the server with hot cache and performance counters,
3. inspect `/moe-layer-perf`,
4. start the server with hot cache and `--no-perf`,
5. compare tok/s with the previous run.

## Final Notes For Future Developers

This fork accelerates Qwen3.x MoE decode by caching frequently used experts and running hot GPU work in parallel with cold CPU work.

The normal llama.cpp path remains active as long as no hot-cache budget is set.

The experimental code was moved into dedicated files:

- graph specialization in `llama-moe-hot-cache-graph.cpp`,
- cache construction in `llama-moe-hot-cache.cpp`,
- performance collection in `llama-moe-hot-cache-perf.cpp`,
- scheduler extension in `ggml-backend-moe-hot-cache.inc`,
- experimental scheduler API in `ggml-backend-moe-hot-cache.h`.

Some core hooks remain necessary, especially in the scheduler and `MUL_MAT_ID`. Keep them small and review them carefully during upstream updates.

The observed performance moved from about 16 to 17 tok/s in early programming-task runs to about 28.09 tok/s in the latest referenced run. The next major gains are likely to come from lower overhead and a faster cold path, not from simply adding more hot cache, because the practical hot-hit rate is expected to stay around 70 percent.
