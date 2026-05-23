# MoE Hot Cache Runtime Switches

This file summarizes the MoE hot-cache environment variables that are relevant in this fork.

Notes:

- `LLAMA_ARG_*` are the normal llama.cpp environment aliases for CLI arguments.
- In `model_config.ini`, use the argument name without the leading `--`, for example `moe-hot-cache-max-mib = -1`.
- The plain `LLAMA_MOE_*` switches are development and runtime levers. They currently have no CLI/INI argument unless the table explicitly says otherwise.
- Boolean switches with default `on` effectively accept: unset/empty/on/true/1 = on, `0`/`off`/`false` = off.

## Argument-Backed Options

| Environment variable | Matching argument / INI key | Options | Default | Effect |
|---|---|---|---|---|
| `LLAMA_ARG_CPU_MOE` | `--cpu-moe` / `cpu-moe = true` | Boolean | off | Practical requirement for the hot-cache path. Keeps the normal MoE expert tensors on CPU/RAM so `--moe-hot-cache` only copies the selected hot experts into VRAM and the cold path stays controlled on CPU. |
| `LLAMA_ARG_MOE_HOT_CACHE` | `--moe-hot-cache FNAME` / `moe-hot-cache = FNAME` | Path to JSON file | unset | Specifies the `/moe-layer-perf` JSON used to build the initial hot cache. Required when `--moe-hot-cache-max-mib` is not `0`. |
| `LLAMA_ARG_MOE_HOT_CACHE_MAX_MIB` | `--moe-hot-cache-max-mib N` / `moe-hot-cache-max-mib = N` | `0`, `>0`, `-1` | `0` | `0` disables the hot cache. Positive values set a fixed MiB budget. `-1` auto-sizes from VRAM remaining after model/KV allocation and requires an explicit `--ctx-size`. |
| `LLAMA_ARG_MOE_HOT_CACHE_AUTO_RESERVE_MIB` | `--moe-hot-cache-auto-reserve-mib N` / `moe-hot-cache-auto-reserve-mib = N` | Integer `>= 0` | `1024` | Used only with `--moe-hot-cache-max-mib -1`. Leaves N MiB free after auto-budgeting for warmup, compute buffers, and CUDA transient allocations. |
| `LLAMA_ARG_MOE_HOT_CACHE_UPDATE_RATE` | `--moe-hot-cache-update-rate N` / `moe-hot-cache-update-rate = N` | Float `0.0` to `1.0` | `0.0` | After completed server runs, replaces up to N fraction of the current hot-cache entries with better candidates. Requires an active hot cache. |
| `LLAMA_ARG_MOE_HOT_CACHE_LAYER_CURVE`; internally also `LLAMA_MOE_HOT_CACHE_LAYER_CURVE` | `--moe-hot-cache-layer-curve N` / `--moe-hot-cache-qwen-layer-curve N` / `moe-hot-cache-layer-curve = N` | Float `0.0` to `1.0` | `0.5` | Gives more weight to layers with higher timing pressure during initial selection and dynamic updates. Primarily for Qwen35Moe; Gemma4 additionally has `LLAMA_MOE_HOT_CACHE_GEMMA4_LAYER_CURVE`. Older specific aliases such as `LLAMA_MOE_HOT_CACHE_QWEN_LAYER_CURVE` are still read as fallbacks. |
| `LLAMA_ARG_MOE_HOT_CACHE_WEIGHTING`; internally also `LLAMA_MOE_HOT_CACHE_WEIGHTING` | `--moe-hot-cache-weighting MODE` / `moe-hot-cache-weighting = MODE` | `flat`, `pressure`, `smooth`, `smooth-pressure`, `capped`, `capped-pressure`, `soft-pressure`, `time`, `moe-time`, `decode-time`, `balanced`, `rank`, `layer-rank` | `flat` | Selects the ranking mode for initial hot-cache selection and dynamic updates. `flat` spreads the budget as evenly as possible over observed layers by sorting experts by hits within each layer and then interleaving equal ranks across layers. `pressure` restores the previous pressure-weighted default. |
| `LLAMA_ARG_MOE_HOT_CACHE_PP_REDUCE_MERGE`; internally also `LLAMA_MOE_HOT_CACHE_PP_REDUCE_MERGE` | `--moe-hot-cache-pp-reduce-merge MODE` / `moe-hot-cache-pp-reduce-merge = MODE` | `off`, `on`, `auto` | `off` | Experimental prompt-processing lever. Reduces the hot and cold branches to `[n_embd, n_tokens]` before both branches are merged. This avoids the large slot merge for batched PP. `auto` enables the path only for larger PP batches. Decode is unchanged. |
| `LLAMA_ARG_MOE_LAYER_PERF_OUT` | `--moe-layer-perf-out FNAME` / `moe-layer-perf-out = FNAME` | Path to JSON file | unset | Server helper for the first profiling run. Enables perf/expert counts and writes the current `/moe-layer-perf` JSON after completed requests and during shutdown. |

## Env-Only Runtime And Development Levers

| Environment variable | Matching argument / INI key | Options | Default | Effect |
|---|---|---|---|---|
| `LLAMA_MOE_LAYER_PERF` | No direct CLI argument. Runtime switch via `POST /moe-layer-perf` with `{"mode":"full"}`, `{"mode":"update"}`, or `{"mode":"off"}`. `--no-perf` starts initially in `off`. | `full`, `update`, `off`; aliases: `1`/`on`/`true` for `full`, `0`/`false` for `off` | `full`, except with `--no-perf`: `off` | Controls which MoE perf counters are active. `full` collects all timing and expert data, `update` only the data needed for dynamic update / hit rate, and `off` disables the path. |
| `LLAMA_MOE_HOT_CACHE_PARALLEL` | No argument | unset/empty/`1`/`on`/`true`/`auto` = auto, `0`/`off`/`false` = off, `force` = force | Auto | Enables the hot/cold fork-join region in the scheduler. Auto falls back to serial execution for unsuitable regions; `force` turns that into an error and is intended for debugging. |
| `LLAMA_MOE_HOT_CACHE_PARALLEL_MIN_SLOTS` | No argument | Integer `>= 0` | `2` | Minimum number of hot+cold slots before auto parallelization starts. `0` means always try the parallel region. |
| `LLAMA_MOE_HOT_CACHE_GEMMA4_LAYER_CURVE` | No argument | Float `0.0` to `1.0` | `0.5` | Gemma4-specific layer-pressure weighting for initial selection and dynamic update. Conceptually the same as the Qwen curve, but without a CLI/INI argument yet. |
| `LLAMA_MOE_HOT_CACHE_BRANCH_REDUCE_MERGE` | No argument | Boolean | on | Enables the Branch-Reduce-Merge path as a Gemma4-specific comparison/fallback lever. In normal decode, direct decode merge is now the primary Gemma4 path; Branch-Reduce-Merge matters when direct decode merge is disabled. Qwen35Moe explicitly disables this profile switch. |
| `LLAMA_MOE_HOT_CACHE_MERGE_SUM_ROWS` | No argument | Boolean | on | Uses an optimized sum path to reduce multiple MoE slot outputs into the final token result. |
| `LLAMA_MOE_HOT_CACHE_CPU_DECODE_ROUTING` | No argument | Boolean | on | Moves hot/cold routing and worklist creation in decode to a CPU custom-op path. |
| `LLAMA_MOE_HOT_CACHE_DECODE_DIRECT_MERGE` | No argument | Boolean | on, but architecture-dependent | Decode optimization for Qwen35Moe and Gemma4: merges single-token decode directly into the final result instead of first creating larger slot intermediates. |
| `LLAMA_MOE_HOT_CACHE_DECODE_STRIDED_SUM_ROWS` | No argument | Boolean | on | Optimized sum path for decode merge when multiple slot rows must be reduced. |
| `LLAMA_MOE_HOT_CACHE_HOT_DUMMY_PADDING` | No argument | Boolean | on | Adds dummy hot work so graph shape, worklist, and backend scheduler stay stable for empty/small hot lanes. |
| `LLAMA_MOE_HOT_CACHE_SHARED_INPUT_ROW` | No argument | Boolean | on | Allows decode to share the same input row for cold work when all cold experts use the same token input. |
| `LLAMA_MOE_HOT_CACHE_COLD_PREFIX_SUM` | No argument | Boolean | on, but architecture-dependent | Cold decode optimization: treats valid cold slots as a compact prefix and reduces only that region. Can be enabled for Qwen35Moe and Gemma4; other architectures need explicit profile approval. |
| `LLAMA_MOE_HOT_CACHE_COLD_PREFIX_WEIGHTED_SUM` | No argument | Boolean | on, but only with `LLAMA_MOE_HOT_CACHE_COLD_PREFIX_SUM` | Folds expert weights directly into the cold-prefix sum and saves separate weighting/merge work. |
| `LLAMA_MOE_HOT_CACHE_DECODE_REPEAT_HOT_INPUT` | No argument | Boolean | on | Repeats the current decode token for hot work directly instead of fetching it through a separate gather path. |
| `LLAMA_MOE_HOT_CACHE_COLD_FIRST_ROW_INPUT` | No argument | Boolean | on | Together with `LLAMA_MOE_HOT_CACHE_SHARED_INPUT_ROW`, uses the first input row as the shared cold input and reduces gather/input overhead. |

## Obsolete Or Discarded Switches

These names appear in older notes or discarded experiments, but have no active code effect in the current branch:

| Environment variable | Replacement / status | Reason |
|---|---|---|
| `LLAMA_MOE_HOT_CACHE_JSON` | Replaced by `--moe-hot-cache` or `LLAMA_ARG_MOE_HOT_CACHE` | Old local starter name for the hot-cache JSON. |
| `LLAMA_MOE_HOT_CACHE_MAX_MIB` | Replaced by `--moe-hot-cache-max-mib` or `LLAMA_ARG_MOE_HOT_CACHE_MAX_MIB` | Old local starter name for the cache budget. |
| `LLAMA_MOE_HOT_CACHE_QWEN_GPU_COLD_MERGE` | Discarded | Experiment moved merge work to GPU, but increased cold-lane/sync pressure and was slower. |
| `LLAMA_MOE_HOT_CACHE_QWEN_COLD_PREFIX_TASKS` | Discarded | Experiment split the CPU prefix sum into multiple tasks. 2 and 4 tasks were slower than the single-task default. |
