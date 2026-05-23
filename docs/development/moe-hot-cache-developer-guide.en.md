# MoE Hot Cache: Developer Guide

Status: 2026-05-22
Branch: `cached-experts-v2`  
Current working state: `flat` is the default hot-cache weighting mode; the hot-cache path expects `--cpu-moe`.

This document is the English developer guide for the experimental MoE hot-cache work in this fork. It explains what was changed, why it was changed, how the runtime path works, which switches matter, how to interpret the performance data, and where the remaining maintenance risks are.

The implementation targets Qwen3.5/Qwen3.6 MoE decode throughput. The core idea is to cache frequently used experts on the GPU and run the remaining cold expert work on the CPU in parallel. There is also a separate experimental hot-graph hook for Gemma 4 26B-A4B.

Important Gemma4 naming note: according to the local GGUF metadata, `gemma-4-E4B` is a dense Gemma4 model with 42 transformer layers, not an MoE model. The relevant Gemma variant for MoE hot-cache work is `gemma-4-26B-A4B`, meaning 26B total parameters with roughly 4B active parameters.

## Summary

The change adds an optional hot cache for MoE experts.

Without the hot cache, the server should behave like the previous llama.cpp path. The hot-cache path is designed for `--cpu-moe`: normal MoE expert tensors stay on the CPU/RAM path, while only selected hot experts are copied into the GPU cache. The experimental path only becomes active when a non-zero hot-cache budget is configured:

```text
--moe-hot-cache-max-mib != 0
```

When a budget is set, a hot-cache JSON file is required as well:

```text
--moe-hot-cache <file.json>
```

The hot-cache path:

1. reads a performance JSON with layer and expert usage data,
2. selects hot experts per layer within a memory budget,
3. copies the selected experts into dedicated hot-cache tensors,
4. builds a specialized MoE hot graph for active layers of the current model,
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
| `flat` control run with the normal `qwen36` profile, 1024 tokens | about 24.13 tok/s at 53.27 percent hit rate; practically tied with `pressure` in the same short run |
| Qwen3.6 break-even series | standard llama in router mode was about 22.2 tok/s; hot cache crossed it at about 45.9 percent real hit rate |
| Qwen3.6 with MTP | no benefit on the local setup; one MTP run reached about 25.33 tok/s despite about 94 percent draft acceptance, below good non-MTP hot-cache runs |
| Gemma4 26B-A4B after decode/merge changes | more stable than the first Gemma4 hot graph; direct decode merge and cold-prefix paths gave a visible speedup without touching the Qwen path |
| Quadro M1200 as secondary warm lane | no stable gain; the best measured setup was hot on CUDA0 plus cold on CPU, without the warm lane |

The main gain did not come from one single patch. It came from the combination of:

- lower decode routing overhead,
- lower merge overhead,
- fewer unnecessary branch launches,
- a faster cold path,
- explicit scheduler regions for hot/cold work,
- reduced performance counter overhead,
- strict gating so the normal path remains intact.

The current default for new hot-cache runs is `--moe-hot-cache-weighting flat`. This mode spreads the budget as evenly as possible across observed layers. The previous pressure-weighted mode remains available with `--moe-hot-cache-weighting pressure`.

## Current Experimental Learnings

### `--cpu-moe` Is Part Of The Hot-Cache Model

The hot cache does not optimize an arbitrary GPU offload layout. It optimizes this specific split:

```text
hot experts  -> GPU hot cache
cold experts -> normal CPU MoE path
```

Therefore `--cpu-moe` belongs in the learning run, the normal hot-cache start, and final `--no-perf` measurements. Without `--cpu-moe`, llama.cpp offload rules may already place experts on the GPU; then the controlled cold path expected by the hot/cold graph is missing.

### Full Layer Overrides Were Not Automatically Better

Several Qwen experiments placed weak or early full layers on the GPU with `override-tensor`, then filled the remaining VRAM with hot experts. This looked plausible but was worse than the pure hot-cache approach:

- the hot cache shrinks because full layers consume a lot of VRAM,
- the hit rate of the remaining layers drops,
- fully GPU-resident layers still pass through the hot/cold graph unless a dedicated bypass exists,
- merge and scheduler overhead therefore remain partly visible.

The consequence: full-layer overrides become interesting again only when there is a separate graph path that handles fully GPU-resident MoE layers without the hot/cold split.

### MTP Was Not A Default Win Locally

Qwen3.6 MTP reached high draft acceptance but created additional context, graph, and compute memory pressure. In the observed runs, hot cache without MTP was faster. MTP may be worth retesting with more VRAM or a different llama.cpp MTP implementation, but it is not the recommended default for this branch. Details are in `docs/development/moe-hot-cache-mtp-learnings.en.md`.

### A Second Slow GPU Is Not A Free Cold Replacement

The Quadro M1200 warm-lane experiment showed that CUDA1 does not simply replace CPU work. The warm lane adds synchronization, bridge work, and transfer pressure back to CUDA0, where the final join/merge happens. The best measured Gemma4 setup for this hardware was therefore:

```text
CUDA0: hot cache
CPU:   cold branch
CUDA1: no warm lane
```

Detailed numbers are in `docs/development/moe-hot-cache-warm-lane-analysis.md`.

### Larger RAM-Heavy Models Need Cold-Path Work

In the 122B Qwen test, expert lists changed more strongly and the model lived much more in RAM. In that situation, a finer static expert list is less useful than reducing overhead and cold-path cost. For very large MoE models, the key question is not only "which experts are hot?", but "how cheap is the unavoidable CPU fraction?".

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
- wired-in models use their normal MoE/FFN path,
- the hot-cache graph is not built,
- the scheduler does not see a hot-cache parallel region.

The new path is active when both are provided:

```text
--cpu-moe
--moe-hot-cache-max-mib <N>
--moe-hot-cache <file.json>
```

`N > 0` builds a cache with a fixed MiB budget. `N = -1` enables auto-sizing: the cache is built later, after the real KV cache allocation in `llama_context`, and uses the then remaining VRAM minus `--moe-hot-cache-auto-reserve-mib`.

If a budget is set but no JSON file is provided, that is intentionally treated as an error. Otherwise the runtime would not know which experts should be cached. `--cpu-moe` is not a hard parser dependency, but it is a functional requirement for the intended hot/cold path.

The model-specific hot path is built per layer only when:

```text
llama_moe_hot_cache_layer_active(model, il)
```

returns true.

This allows some layers to use the hot path while other layers still use the normal path.

## Runtime Switches

### MoE Placement

```text
--cpu-moe
LLAMA_ARG_CPU_MOE=true
```

Keeps all MoE expert weights on the CPU/RAM path. For the hot cache this is the recommended and practically required starting state: cold experts remain on the CPU, while hot experts are copied into the GPU hot cache as an additional copy.

`--n-cpu-moe` is not equivalent. It only moves the first N MoE layers to the CPU and does not create a clean full cold path for the hot cache.

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

Maximum memory budget for the hot cache in MiB. `0` disables the feature, positive values are fixed budgets, and `-1` enables auto-sizing. Auto-sizing requires an explicit `--ctx-size`, because the KV cache is part of the memory decision.

```text
--moe-hot-cache-auto-reserve-mib <N>
LLAMA_ARG_MOE_HOT_CACHE_AUTO_RESERVE_MIB=<N>
```

Only relevant with `--moe-hot-cache-max-mib -1`. The value controls how many MiB remain free after KV cache allocation and before the hot cache is allocated. The default is `1024`. Higher values are more conservative and avoid CUDA OOM during warmup or compute transients; lower values make the hot cache larger.

```text
--moe-hot-cache-qwen-layer-curve <N>
LLAMA_ARG_MOE_HOT_CACHE_LAYER_CURVE=<N>
LLAMA_MOE_HOT_CACHE_LAYER_CURVE=<N>
```

Primarily Qwen35Moe. Controls the layer-pressure weighting curve for hot-cache selection. `0.0` uses the normal expert ranking without layer wait time. `0.5` is the damped default. `1.0` weights waiting layers aggressively. Internally the weighting first reads `LLAMA_MOE_HOT_CACHE_LAYER_CURVE`; older specific aliases such as `LLAMA_MOE_HOT_CACHE_QWEN_LAYER_CURVE` are still accepted as fallbacks.

```text
--moe-hot-cache-weighting <MODE>
LLAMA_ARG_MOE_HOT_CACHE_WEIGHTING=<MODE>
```

Controls the ranking mode for initial hot-cache selection and dynamic updates. The most relevant values are `flat`, `pressure`, `smooth`, `time`, and `balanced`; additional development variants include `smooth-pressure`, `capped`, `capped-pressure`, `soft-pressure`, `moe-time`, `decode-time`, `rank`, and `layer-rank`. The default is `flat`. `flat` spreads the budget as evenly as possible over the observed layers: experts are ranked by hits inside each layer, then equal ranks are interleaved across layers. The layer curve has no effect in `flat` mode. Use `pressure` to restore the previous pressure-weighted default.

```text
--moe-hot-cache-pp-reduce-merge <off|on|auto>
LLAMA_ARG_MOE_HOT_CACHE_PP_REDUCE_MERGE=<off|on|auto>
LLAMA_MOE_HOT_CACHE_PP_REDUCE_MERGE=<off|on|auto>
```

Experimental prompt-processing lever for hot-cache runs. The default is `off`. During prompt processing, physical `ubatch` graphs often contain hundreds of tokens. Without this lever, the hot and cold branches are merged while they are still expert-slot tensors and are reduced to `[n_embd, n_tokens]` afterward. With `on` or `auto`, each branch first reduces its own slots to `[n_embd, n_tokens]`; only the smaller hot-plus-cold result is merged afterward. `auto` enables this path only for larger PP batches. Decode (`n_tokens == 1`) is unchanged. The operation is mathematically equivalent, but it changes the floating-point addition order and can therefore slightly affect sampled text.

```text
LLAMA_MOE_HOT_CACHE_GEMMA4_LAYER_CURVE=<N>
```

Gemma4 only. Controls the same layer-pressure weighting for initial hot-cache selection and dynamic updates. `0.0` uses the normal expert ranking without layer pressure, `0.5` is the default, and `1.0` weights waiting layers aggressively. This switch is currently environment-variable only.

### Parallelization

```text
LLAMA_MOE_HOT_CACHE_PARALLEL=1
```

Enables hot/cold parallelization in auto mode. This is also the default when `LLAMA_MOE_HOT_CACHE_PARALLEL` is not set.

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

Minimum number of slots required before the scheduler launches the hot/cold parallel region. Default: `2`.

This avoids paying thread and scheduler overhead for tiny work packets.

```text
LLAMA_MOE_HOT_CACHE_BRANCH_REDUCE_MERGE=0
```

Disables the Gemma4 Branch-Reduce-Merge comparison path. The primary Gemma4 decode path is now direct decode merge with a compact cold prefix. Branch-Reduce-Merge remains useful for comparison runs or when `LLAMA_MOE_HOT_CACHE_DECODE_DIRECT_MERGE=0` is set. Qwen35Moe explicitly disables this profile switch.

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

Disables llama.cpp performance counters and starts the MoE performance mode as `off`. The mode can still be changed to `update` or `full` at runtime.

```text
--moe-layer-perf-out <file.json>
LLAMA_ARG_MOE_LAYER_PERF_OUT=<file.json>
```

Server-only first-run profiling helper. It enables detailed expert counts and writes the current `/moe-layer-perf` JSON to the given file after completed requests and once more during shutdown. Without an active hot cache, this produces raw per-layer `experts` lists. With an active hot cache, it can also produce `hot_experts` and `cold_experts`.

```text
LLAMA_MOE_LAYER_PERF=full|update|off
```

Sets the initial MoE performance mode when the server does not derive it from `--no-perf`. `full` collects all existing counters and timing fields. `update` collects only the data needed by dynamic hot-cache updates: expert counts, hot/cold slots, and hot/cold/join wait timings. `off` disables the MoE performance path.

The mode can be changed at runtime via HTTP:

```bash
curl -X POST http://127.0.0.1:8080/moe-layer-perf \
  -H 'Content-Type: application/json' \
  -d '{"mode":"update"}'
```

In router mode, pass `?model=<name>&autoload=false`.

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
- neutral scoring for non-specialized models,
- selecting experts within the memory budget,
- allocating hot-cache tensors,
- copying selected experts,
- building mapping and mask tables.

Supported schemas:

```text
llama.cpp.moe_layer_perf.v1
llama.cpp.moe_layer_opt_perf.v1
```

The generic selection intentionally stays simple. It uses observed expert counts without model-specific layer heuristics. Qwen35Moe and Gemma4 have dedicated weighting classes that turn the same observation list into architecture-specific scores.

### `src/models/qwen35moe-hot-cache.cpp`

Contains the Qwen35Moe-specific weighting for hot-cache selection.

The `llama_moe_hot_cache_qwen35moe_weighting` class rescores the neutral layer/expert observations. For Qwen35Moe, the score is not only based on how often an expert appears, but also on how strongly the layer stalls the parallel region. The weighting prefers absolute join wait per layer and falls back to cold/hot lane delta, cold slots, or wait per cold slot.

The curve strength is controlled by `--moe-hot-cache-qwen-layer-curve`. The default `0.5` is intentionally damped: waiting layers are favored, but other layers should not be pushed out of the cache too aggressively.

The same weighting is also used by dynamic updates. During updates, it can only prioritize exchange candidates between layers. The number of hot-cache slots per layer remains unchanged, because real redistribution would require tensor reallocation or a second cache.

Experts that are already hot get a small sticky bonus. This keeps the cache set more stable and avoids churn from small one-off changes.

### `src/models/gemma4-hot-cache.cpp`

Contains the Gemma4-specific weighting for hot-cache selection.

The `llama_moe_hot_cache_gemma4_weighting` class rescoring uses the same layer/expert observations as the Qwen weighting, but stays separate from the Qwen code. The weighting prefers absolute join wait per layer and falls back to cold/hot lane delta, cold slots, or wait per cold slot.

The curve strength is controlled by `LLAMA_MOE_HOT_CACHE_GEMMA4_LAYER_CURVE`. The default is `0.5`. The weighting is used both during initial JSON loading and dynamic updates. Experts that are already hot also get a small sticky bonus.

### `src/llama-moe-hot-cache-graph.cpp`

Contains the extracted hot-cache graph logic for the wired-in models.

This was an important refactor. The experimental graph logic was moved out of the model files, so `src/models/qwen35moe.cpp` and `src/models/gemma4.cpp` stay closer to upstream.

Responsibilities:

- build the hot/cold FFN for Qwen35Moe,
- build the Gemma4 hot/cold MoE from logits,
- build decode-specific worklists,
- wire CPU custom ops for routing and merge,
- build the hot branch using cached experts,
- build the cold branch using normal experts,
- merge both branch outputs,
- annotate the scheduler parallel region,
- set local `mul_mat_id` flags.

This code is not a generic MoE replacement for all architectures. New models should be added through small model hooks, architecture-specific profiles, and dedicated weighting classes so the Qwen paths do not get side effects.

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

### `src/models/gemma4.cpp`

Contains only the small Gemma4 hook into the extracted hot graph:

```text
if the layer has an active hot cache:
    build_layer_moe_hot(...)
else:
    normal build_moe_ffn(...)
```

The actual Gemma4 hot-cache logic stays in `src/llama-moe-hot-cache-graph.cpp` and `src/models/gemma4-hot-cache.cpp`. This keeps the Gemma4 model path separate from the Qwen path.

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
- expert counts in `full` and `update` mode,
- temporary scheduler split debug data in `full` mode.

`parallel_split_debug` shows the last observed hot, cold, and join splits including backend IDs. It is only diagnostic for scheduler work and is not needed by dynamic updates.

When performance collection is disabled, the JSON function intentionally returns only a small disabled block:

```json
{
  "enabled": false,
  "mode": "off",
  "schema": "llama.cpp.moe_layer_opt_perf.v1",
  "layers": []
}
```

### `src/llama-context.cpp`

Contains small hooks to:

- build the auto hot cache for `--moe-hot-cache-max-mib -1` after the real KV cache allocation,
- set the performance callback,
- enable scheduler performance collection,
- compute the graph as before.

When the MoE performance mode is `off`, the MoE performance path is not installed. `--no-perf` sets this mode to `off` at server startup.

### `src/llama.cpp`

Initializes the hot cache during model loading for fixed budgets:

```text
llama_moe_hot_cache_init(*model, params)
```

The function is gated by `moe_hot_cache_max_mib`. Positive budgets are built during model loading. `-1` is intentionally deferred until context creation so the implementation can use the real free VRAM after KV cache allocation.

### `src/llama-model.cpp` And Related Parameter Files

Extend model parameters with:

```text
moe_hot_cache_path
moe_hot_cache_max_mib
moe_hot_cache_auto_reserve_mib
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
- optionally allow a bridge split between lane end and join,
- run a CPU worker for the cold path,
- let the normal thread/GPU backend run the hot path,
- join both paths,
- collect errors and fallback reasons,
- collect scheduler performance data.

The bridge split is allowed through `GGML_BACKEND_SCHED_MOE_HOT_CACHE_PARALLEL_FLAG_ALLOW_JOIN_BRIDGE`. This matters for Gemma4 because cold-prefix merge or Branch-Reduce-Merge can create a small serial split between the hot/cold lanes and the join. Without the flag, validation stays strict at hot-then-cold-then-join.

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
- gating cases,
- Qwen and Gemma4 weighting.

Future decode-path optimizations should extend this test file when possible.

## Runtime Flow

### 1. Startup Parameters Are Parsed

The CLI/common parameter layer reads the hot-cache arguments and performance switches.

Important:

```text
--moe-hot-cache-max-mib 0
```

is the default and means: no hot cache.

### 2. The Model And Context Are Loaded

When `moe_hot_cache_max_mib > 0`, the hot cache is initialized during model loading with a fixed budget.

When `moe_hot_cache_max_mib == -1`, the hot cache is built inside `llama_context` after the real KV cache allocation has completed. The implementation then reads remaining VRAM, subtracts `moe_hot_cache_auto_reserve_mib`, and uses the rest as the hot-cache budget.

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

The root object always contains `mode`. The relevant modes are:

| Mode | JSON content | Purpose |
| --- | --- | --- |
| `full` | All existing counters, expert lists, and timing fields | Detailed analysis and UI timings |
| `update` | Expert lists, hot/cold slots, hit rate, hot lane, cold lane, join wait | Dynamic hot-cache updates with lower measurement overhead |
| `off` | `enabled=false`, no layer data | Throughput tests without MoE perf overhead |

Important `full` fields:

```text
summary.layer_calls
summary.hot_slot_ratio
summary.total_moe_time_per_call_us
summary.routing_time_per_call_us
summary.worklist_time_per_call_us
summary.merge_time_per_call_us
summary.hot_branch_time_per_call_us
summary.cold_branch_time_per_call_us
summary.hot_expert_matmul_time_per_call_us
summary.cold_expert_matmul_time_per_call_us
summary.hot_gather_scatter_time_per_call_us
summary.cold_gather_scatter_time_per_call_us
summary.parallel_region_wall_time_per_call_us
summary.parallel_hot_lane_wall_time_per_call_us
summary.parallel_cold_lane_wall_time_per_call_us
summary.parallel_join_wait_time_per_call_us
summary.parallel_overlap_estimate_per_call_us
summary.parallel_hot_launches
summary.parallel_cold_launches
summary.parallel_fallbacks
layers[]
```

`update` reduces this to the fields required by `llama_moe_hot_cache_update_from_perf_json(...)` and the hit-rate view:

```text
summary.layer_calls
summary.hot_slot_ratio
summary.parallel_hot_lane_wall_time_per_call_us
summary.parallel_cold_lane_wall_time_per_call_us
summary.parallel_join_wait_time_per_call_us
layers[].calls
layers[].hot_slots_total
layers[].cold_slots_total
layers[].hot_slots_per_call
layers[].cold_slots_per_call
layers[].hot_slot_ratio
layers[].parallel_hot_lane_wall_time_per_call_us
layers[].parallel_cold_lane_wall_time_per_call_us
layers[].parallel_join_wait_time_per_call_us
layers[].experts
layers[].hot_experts
layers[].cold_experts
```

Expert counters are included in `full` and `update`. In `off`, they are not collected.

## Live Visualization In The Web UI

The Web UI has a dedicated MoE layer performance page:

```text
#/moe-layer-perf
```

In chat, open it with the activity button next to the input actions. The button is intentionally not placed next to the live `t/s` statistic because that area can move while generation is still running. A mode dropdown next to the button switches between `Full`, `Update`, and `Off`.

`Full` returns all existing counters and timing fields. `Update` returns only the data needed for `--moe-hot-cache-update-rate` and the hit-rate view. `Off` disables the MoE performance path. If the server was started with `--no-perf`, the dropdown starts in `Off`.

![MoE layer performance UI](assets/moe-layer-perf-overview-wide.png)

The page reads the same data as the `/moe-layer-perf` JSON endpoint. In router mode, it passes the currently selected model as a query parameter and sets `autoload=false`, so merely opening the page does not force a model load or model switch. Mode changes use the same path via `POST /moe-layer-perf`.

The page refreshes automatically. The update interval is configurable in the header and limited to `0.5` to `3.0` seconds. The manual refresh button was removed intentionally because repeated clicking adds endpoint requests and UI work.

The top graph shows hot hit rate by layer. The layer cards below show each expert:

| Color | Meaning |
| --- | --- |
| Red | Hot expert |
| Blue | Cold expert |
| Yellow | Active since the previous UI update |
| Gray | No hit in the current performance window |

Yellow is not read from a separate `active_experts` field. The UI compares the current expert counters with the previous poll. If a counter increased, that expert was active during the last interval.

Blank cells are only square-grid padding when the expert count does not fit an exact square. Normal gray cells are real experts with no hits.

### Timing Order In The UI

The timing cards are ordered by execution flow, not by size:

1. `Summary`
2. `Routing / prep`
3. `Parallel region`
4. `Hot lane` and `Cold lane`
5. `Synchronization`
6. `Merge`

This order makes the pipeline easier to read:

| UI group | Relevant fields | Interpretation |
| --- | --- | --- |
| `Summary` | `total_moe_time_per_call_us`, `layer_calls`, `parallel_fallbacks` | Top-level state of the current performance window |
| `Routing / prep` | `routing_time_per_call_us`, `worklist_time_per_call_us` | Work before the hot/cold split |
| `Parallel region` | `parallel_region_wall_time_per_call_us` | Wall time of the full parallel scheduler region |
| `Hot lane` | `parallel_hot_lane_wall_time_per_call_us`, `hot_branch_time_per_call_us`, `hot_gather_scatter_time_per_call_us`, `hot_expert_matmul_time_per_call_us`, `parallel_hot_launches` | GPU hot-cache side of the parallel region |
| `Cold lane` | `parallel_cold_lane_wall_time_per_call_us`, `cold_branch_time_per_call_us`, `cold_gather_scatter_time_per_call_us`, `cold_expert_matmul_time_per_call_us`, `parallel_cold_launches` | Cold expert side of the parallel region |
| `Synchronization` | `parallel_overlap_estimate_per_call_us`, `parallel_join_wait_time_per_call_us` | How much work overlapped and how long one lane waited at the join |
| `Merge` | `merge_time_per_call_us` | Merge of hot and cold outputs |

Important: these values are not all additive. `Parallel wall` is the wall time of the parallel region. `Hot lane` and `Cold lane` are lane wall times inside that region. `Hot branch` and `Cold branch` are internal sub-measurements and should not simply be added to lane wall time.

`Overlap` is the estimated time hidden by parallel execution. `Join wait` is the time the first finished lane waits at the join point. In the current Qwen runs, the hot lane usually waits for the slower cold lane, but the opposite is possible in principle.

## Creating The First Hot-Cache JSON

If no hot-cache JSON exists yet, start without the hot cache but with expert counts enabled:

```bash
./build/bin/llama-server \
  --cpu-moe \
  --moe-layer-perf-out moe-hot-cache.json \
  <normal model and server arguments>
```

Then run representative prompts. The output file is updated after completed requests and once more during shutdown. The same data can still be inspected through the existing `/moe-layer-perf` endpoint. The first profile is built from the raw `experts` arrays; later profiles can use `hot_experts` and `cold_experts` when the hot cache is already active.

The recommended first-run workflow is `--moe-layer-perf-out <file.json>`. For normal dynamic updates after that, the slimmer `update` mode is enough.

That JSON can then be used as input for the hot cache:

```bash
./build/bin/llama-server \
  --cpu-moe \
  --moe-hot-cache performance.json \
  --moe-hot-cache-max-mib -1 \
  --moe-hot-cache-auto-reserve-mib 1024 \
  --moe-hot-cache-update-rate 0.10 \
  --moe-hot-cache-qwen-layer-curve 0.5 \
  <normal model and server arguments>
```

`--moe-hot-cache-update-rate` is optional. `0.0` disables dynamic updates; `0.10` replaces up to 10 percent of the current hot-cache entries after completed server runs.

`--moe-hot-cache-qwen-layer-curve` only affects Qwen35Moe. The curve is used both during initial JSON loading and dynamic updates. `0.0` means no layer-pressure weighting, `0.5` is the damped default, and `1.0` weights waiting layers aggressively. In the default `flat` mode, this curve is ignored.

The even layer distribution is now the default:

```bash
--moe-hot-cache-weighting flat
```

`flat` sorts experts by hits within each layer and then interleaves equal ranks across layers. This gives each observed layer roughly the same number of experts for the same budget. The argument can be omitted because `flat` is the default. Use `--moe-hot-cache-weighting pressure` to restore the previous behavior.

For Gemma4, the same mechanism is exposed as an environment variable:

```bash
LLAMA_MOE_HOT_CACHE_GEMMA4_LAYER_CURVE=0.5
```

Gemma4 currently primarily uses direct decode merge with a compact cold prefix. `LLAMA_MOE_HOT_CACHE_BRANCH_REDUCE_MERGE` remains available as a Gemma4-specific comparison/fallback lever, but during decode it only matters when direct decode merge is disabled. Qwen35Moe does not use Branch-Reduce-Merge.

For long prompts, prompt processing with hot/cold splitting can become relevant on its own. The PP-specific reduce-merge lever can be tested with:

```bash
--moe-hot-cache-pp-reduce-merge auto
```

With `auto`, decode remains unchanged. For large PP `ubatch` graphs, the hot branch and cold branch first reduce their expert slots to `[n_embd, n_tokens]`; only then are both branch results added. This avoids a large slot merge. In local tests, Gemma4 PP improved from `87.57 tok/s` to `100.74 tok/s` on a long prompt; Qwen35Moe improved from `71.21 tok/s` to `75.61 tok/s`. These numbers are hardware- and cache-list-dependent; the useful validation metrics are `prompt eval time`, prompt tokens/s, `parallel_fallbacks == 0`, and unchanged or better TG rate.

For final throughput measurements:

```bash
./build/bin/llama-server \
  --no-perf \
  --cpu-moe \
  --moe-hot-cache performance.json \
  --moe-hot-cache-max-mib -1 \
  --moe-hot-cache-auto-reserve-mib 1024 \
  --moe-hot-cache-update-rate 0.10 \
  --moe-hot-cache-qwen-layer-curve 0.5 \
  <normal model and server arguments>
```

## Why `--no-perf` Matters

Performance counters are not free.

Even when the JSON endpoint is queried rarely, decode still has to update measurement points, counters, and sometimes scheduler data. For very small decode steps, this overhead is visible.

There are therefore three useful MoE performance modes.

Development mode:

```text
full
```

Goal: understand where time is spent. This mode collects all timing fields.

Update mode:

```text
update
```

Goal: dynamic hot-cache updates with lower measurement overhead. This mode collects expert counts, slot counts, and hot/cold/join wait timings, but it avoids the full per-node timing callback.

Speed mode:

```text
--no-perf
off
```

Goal: measure real throughput without measurement overhead. Dynamic updates need `update` or `full`; in `off`, the cache is not adapted from new performance data.

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

### 5. Stabilized Gemma4 Decode

Gemma4 needed its own profile class even though many building blocks are shared with Qwen. The reason is not only expert selection; it is graph shape. Gemma4 builds the MoE path from router logits and has different split and merge behavior than Qwen35Moe.

The useful recent Gemma4 levers were:

- a small hook in `src/models/gemma4.cpp`, with the real logic extracted to `src/llama-moe-hot-cache-graph.cpp`,
- separate weighting in `src/models/gemma4-hot-cache.cpp`,
- direct decode merge as the primary path,
- cold-prefix sum and weighted cold-prefix sum so only valid cold slots are reduced,
- `merge_sum_rows` and strided sum rows for compact slot reduction,
- Branch-Reduce-Merge as a separate Gemma4 lever for comparison runs,
- the scheduler bridge flag for cases where cold-prefix or Branch-Reduce creates a small serial bridge split before the join.

The most important debugging lesson: when Gemma4 fallbacks increase, the problem is usually not hit rate, but graph split shape or an optimization that violates the expected hot-then-cold-then-join region. This is why Gemma4 shortcuts stay in a separate profile instead of being folded into the Qwen path.

### 6. Minimal Performance Path

The performance output was changed from visualization-oriented data to optimization-oriented data.

It remains useful in development mode but causes less overhead. For final measurements it can be disabled completely.

### 7. Refactor For Isolation

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
./build/bin/llama-server \
  --perf \
  --cpu-moe \
  --moe-hot-cache performance.json \
  --moe-hot-cache-max-mib -1 \
  --moe-hot-cache-auto-reserve-mib 1024 \
  <normal arguments>
```

### Server With Hot Cache And No Performance Counters

```bash
./build/bin/llama-server \
  --no-perf \
  --cpu-moe \
  --moe-hot-cache performance.json \
  --moe-hot-cache-max-mib -1 \
  --moe-hot-cache-auto-reserve-mib 1024 \
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
3. Is `parallel_join_wait_time_per_call_us` high?
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

### Dynamic Cache

The hot cache is built at startup from a JSON file. With `--moe-hot-cache-update-rate`, it can be partially updated after completed server runs.

The dynamic update replaces existing cache slots with better matching experts. It does not change the number of hot-cache slots per layer.

True dynamic reallocation is possible but significantly more complex:

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

### Model Specificity

The hot graph is active only for explicitly wired-in models. Qwen35Moe and Gemma4 have separate model hooks, profiles, and weighting classes. Other MoE models use the normal path unless they are explicitly wired in.

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

1. test with `LLAMA_MOE_HOT_CACHE_PARALLEL=0`,
2. use auto mode instead of `force`,
3. increase `LLAMA_MOE_HOT_CACHE_PARALLEL_MIN_SLOTS`,
4. test `--no-perf`,
5. inspect performance JSON fallback counts.

## Rules For Further Optimization

### 1. Measure First

Every new optimization should compare at least:

- final tok/s after a longer run,
- `hot_slot_ratio`,
- `parallel_join_wait_time_per_call_us`,
- `parallel_overlap_estimate_per_call_us`,
- `cold_branch_time_per_call_us`,
- `merge_time_per_call_us`,
- `routing_time_per_call_us`,
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
cmake --build build -j8
./build/bin/test-moe-hot-cache
```

If runtime behavior changed:

1. start the server without hot cache and test a simple inference,
2. start the server with hot cache and performance counters,
3. inspect `/moe-layer-perf`,
4. start the server with hot cache and `--no-perf`,
5. compare tok/s with the previous run.

## Final Notes For Future Developers

This fork accelerates Qwen3.x MoE decode and experimentally Gemma4 decode by caching frequently used experts and running hot GPU work in parallel with cold CPU work.

The normal llama.cpp path remains active as long as no hot-cache budget is set.

The experimental code was moved into dedicated files:

- graph specialization in `llama-moe-hot-cache-graph.cpp`,
- cache construction in `llama-moe-hot-cache.cpp`,
- performance collection in `llama-moe-hot-cache-perf.cpp`,
- Qwen weighting in `src/models/qwen35moe-hot-cache.cpp`,
- Gemma4 weighting in `src/models/gemma4-hot-cache.cpp`,
- scheduler extension in `ggml-backend-moe-hot-cache.inc`,
- experimental scheduler API in `ggml-backend-moe-hot-cache.h`.

Some core hooks remain necessary, especially in the scheduler and `MUL_MAT_ID`. Keep them small and review them carefully during upstream updates.

The observed performance moved from about 16 to 17 tok/s in early programming-task runs to about 28.09 tok/s in the latest referenced run. The next major gains are likely to come from lower overhead and a faster cold path, not from simply adding more hot cache, because the practical hot-hit rate is expected to stay around 70 percent.
