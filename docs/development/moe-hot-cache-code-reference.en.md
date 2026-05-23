# MoE Hot-Cache Code Reference

State: 2026-05-22

This file describes the MoE hot-cache implementation introduced by our own
commits on `cached-experts-v2`. It is intentionally close to the code: what each
method does, why it exists, and how it helps performance or maintainability.

Covered own commits:

| Commit | Topic |
| --- | --- |
| `25c231874` | MoE hot-cache runtime, graph, scheduler extension, perf JSON, server API, tests |
| `ce729f52f` | MoE layer performance UI and runtime perf-mode switch |
| `5a6edbb43` | Workflow and tuning documentation |
| `2b4944906` | Experiment documentation for MTP and Quadro M1200 warm lane |
| `1749fbc83` | Gemma4 cold-prefix-merge optimization |
| `27de9be40` | Current documentation updates after rebase and experiments |

## Architecture

The hot cache is intentionally built as an additional path:

```text
Original llama.cpp model
├── all MoE expert tensors stay in the normal model
│   └── with --cpu-moe, typically on the CPU/RAM path
└── our hot cache
    ├── additionally copies selected expert slices into VRAM
    ├── builds hot_id_map, hot_mask, and cold_mask per layer
    ├── splits inference per token/expert slot into hot and cold worklists
    ├── computes the hot branch on GPU and the cold branch on CPU
    └── merges both outputs back into the normal layer output
```

Important: the hot cache currently does not save RAM. It is an additional VRAM
copy of selected expert slices. This is intentional because the cold lane and
dynamic updates still need access to the original CPU tensors.

## Runtime Flow

1. `--moe-layer-perf-out <file>` or `/moe-layer-perf` creates a perf JSON with
   expert hits, hot/cold slots, and timings.
2. `--moe-hot-cache <file> --moe-hot-cache-max-mib <N|-1> --cpu-moe` reads that
   JSON.
3. `llama_moe_hot_cache_init()` scores experts, computes the budget, copies the
   best expert slices into a dedicated GPU buffer, and sets maps/masks.
4. Qwen35MoE and Gemma4 call the hot graph per layer only when
   `llama_moe_hot_cache_layer_active()` is true.
5. The graph builds a worklist. It splits active Top-K expert slots into hot and
   cold.
6. The scheduler can run hot and cold splits in parallel.
7. After a server request, `--moe-hot-cache-update-rate` can exchange some
   hot-cache slots for currently better experts.

## Data Structures

File: `src/llama-moe-hot-cache.h`

| Name | What it contains | Why it helps |
| --- | --- | --- |
| `llama_moe_hot_cache_entry` | `(layer, expert, hit_count)` after scoring. | Common ranking format for initial fill and tests. |
| `llama_moe_hot_cache_expert_observation` | Raw, hot, and cold hit counters per expert. | Can read old `experts` lists and new `hot_experts`/`cold_experts` at the same time. |
| `llama_moe_hot_cache_layer_observation` | Layer ID, expert list, and timing fields such as join wait, cold lane time, and MoE time. | Foundation for weighting modes that prefer slow layers, not only hits. |
| `llama_moe_hot_cache_expert_size` | Memory cost of one expert slice. | Allows budget selection by MiB instead of simple count. |
| `llama_moe_hot_cache_plan` | Full ranking, selected experts, budget, and used bytes. | Makes cache selection testable and loggable. |
| `llama_moe_hot_cache_update_stats` | Hit rate, candidates, exchange budget, and changed layers. | Provides server logs after dynamic update. |
| `llama_moe_hot_cache_weighting_mode` | `flat`, `pressure`, `smooth_pressure`, `time`, `balanced`. | Separates selection strategy from the cache allocator. |
| `llama_moe_hot_cache_weighting_config` | Weighting mode plus `layer_curve`. | One config type for Qwen, Gemma, and future models. |
| `llama_moe_hot_cache_layer` | Hot-cache tensors, maps, masks, host map, `n_hot`, `n_expert`. | Represents one active hot-cache layer. |
| `llama_moe_hot_cache_layer::active()` | Checks `n_hot`, `hot_id_map`, `hot_mask`, `cold_mask`. | Gate so normal models and uncached layers remain unchanged. |
| `llama_moe_hot_cache_worklist_field` | Field layout of the worklist tensor. | Keeps hot/cold IDs, token IDs, weights, and counts in a compact `ggml_tensor`. |
| `llama_moe_hot_cache` | Vector of all layer caches plus ggml contexts and backend buffers. | Keeps VRAM buffers alive as long as the model lives. |
| `llama_moe_hot_cache::active()` | Checks whether at least one layer is active. | Fast global feature check. |

## Cache Construction And Selection

Files: `src/llama-moe-hot-cache.cpp`, `src/llama.cpp`, `src/llama-context.cpp`

| Method | What it does | How it helps |
| --- | --- | --- |
| `llama_moe_hot_cache_hot_dummy_padding()` | Reads `LLAMA_MOE_HOT_CACHE_HOT_DUMMY_PADDING`; default on. | Negative or invalid hot IDs can be caught through a dummy expert, making CUDA paths more stable. |
| `key(layer, expert)` | Packs layer and expert into a `uint64_t`. | Fast key for maps without a custom pair hash. |
| `tensor_expert_bytes(t)` | Computes `ggml_nbytes(t) / t->ne[2]`. | Gives the cost of one expert slice for budget planning. |
| `add_saturating(dst, value)` | Adds without overflow; saturates on overflow. | Perf counters and hit counters cannot overflow on long runs. |
| `select_gpu_dev(model)` | Uses the first GPU/iGPU from `model.devices`, otherwise the first backend GPU. | Places the cache on the same relevant GPU backend as the model. |
| `new_tensor_like_experts(ctx, src, n_cache, name)` | Creates a 3D hot-cache tensor with the same expert matrix shape but only `n_cache` experts. | Packs selected experts densely into VRAM instead of copying whole layers. |
| `new_tensor_like_scale(ctx, src, n_cache, name)` | Creates an optional scale tensor for quantized/scaled expert tensors. | Keeps quantized expert paths numerically compatible. |
| `zero_tensor(t)` | Fills a hot-cache tensor with zeros. | The dummy expert and unused slots produce deterministic zero. |
| `copy_expert_slice(src, dst, src_expert, dst_expert)` | Copies one expert slice from the original tensor into the hot cache. | This is the actual VRAM cache fill for `up`, `gate`, `gate_up`, `down`. |
| `copy_scale_slice(src, dst, src_expert, dst_expert)` | Copies the matching scale entry. | Prevents wrong scaling for quantized expert slices. |
| `set_tensor_i32_1d()` / `set_tensor_f32_1d()` | Writes individual map/mask values directly into backend tensors. | Dynamic updates can change maps without rebuilding the whole cache. |
| `current_hot_experts(layer)` | Reconstructs from `hot_id_map_host` which original experts are currently cached. | Basis for exchange decisions during dynamic update. |
| `find_or_add_expert_observation(layer, expert)` | Finds an expert counter or creates it. | JSON parsing can merge `experts`, `hot_experts`, and `cold_experts`. |
| `observation_total_hits(layer, expert)` | Uses `raw` or `hot+cold` depending on JSON schema. | Supports learning runs without hot cache and later runs with hot/cold split. |
| `sort_hot_cache_entries(entries)` | Sorts by score descending, then layer and expert. | Deterministic selection, important for reproducible tests. |
| `score_observations_default()` | Scores only by hit count. | Simple fallback and baseline test for perf JSONs. |
| `weighting_config_from_params(params)` | Builds the weighting config from CLI parameters and env fallbacks. | Makes `--moe-hot-cache-weighting` and `--moe-hot-cache-layer-curve` available everywhere. |
| `score_observations_for_arch(arch, observations, params)` | Selects the scoring strategy. Currently Qwen and Gemma use generic weighting. | Extension point for new models without modifying the loader. |
| `collect_expert_sizes(model)` | Collects all expert slices from layers with `ffn_down_exps`. | Prevents non-MoE layers or missing tensors from entering the cache. |
| `estimate_kv_cache_bytes_on_device(model, params, dev)` | Estimates KV-cache memory for `--moe-hot-cache-max-mib -1`. | Auto-sizing can reserve VRAM for KV and avoid warmup OOM. |
| `auto_hot_cache_budget_bytes(model, params, dev, reserve_kv_cache)` | Reads free VRAM, subtracts KV reserve and safety reserve. | `-1` uses as much remaining VRAM as possible, but with a safety margin. |
| `llama_moe_hot_cache_parse_perf_json_observations(json)` | Validates schema, reads layers, expert lists, and timings. | One parser for `/moe-layer-perf` and `--moe-layer-perf-out`. |
| `llama_moe_hot_cache_parse_perf_json(json)` | Returns the default hit ranking. | Backward-compatible, simple entry point for tests and older users. |
| `llama_moe_hot_cache_select(observed, sizes, budget)` | Walks the ranking and packs experts into the budget; each active layer accounts for one dummy expert. | Budget is not exceeded; layers with cache get a safe dummy slot. |
| `llama_moe_hot_cache_init(model, params, reserve_kv_cache)` | Full cache construction: read JSON, score, compute budget, create ggml context, create cache tensors, copy expert slices, set maps/masks. | Central entry point for the feature. All expensive copies happen once at startup or auto init. |
| `llama_moe_hot_cache_init_after_model_load(model, params)` | Builds the cache directly after model load when `max_mib > 0`. | For fixed budgets: cache is created before context/KV. |
| `llama_moe_hot_cache_init_after_context_memory(model)` | Builds the cache after context/KV when `max_mib == -1`. | For auto-sizing: load model/KV first, then use remaining VRAM. |
| `llama_moe_hot_cache_layer_active(model, il)` | Checks whether the layer has an active cache. | Qwen/Gemma can switch between standard and hot graph with a small hook. |
| `llama_model_load_from_file_impl()` hook in `src/llama.cpp` | Calls `llama_moe_hot_cache_init_after_model_load()`. | Fixed cache is initialized without changing model classes. |
| `llama_context` constructor hook | Calls `llama_moe_hot_cache_init_after_context_memory()`. | Auto-cache uses the truly free VRAM after context memory. |

## Worklists

Files: `src/llama-moe-hot-cache.cpp`, `src/llama-moe-hot-cache-graph.cpp`

The worklist is an `F32` tensor with shape:

```text
[capacity, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COUNT]
capacity = n_tokens * n_expert_used
```

Each field is a row. Valid hot slots are densely packed at the start of the hot
fields; valid cold slots are densely packed at the start of the cold fields.

| Method | What it does | How it helps |
| --- | --- | --- |
| `llama_moe_hot_cache_build_worklist(dst, selected_experts, weights, layer, ith, nth)` | Takes completed Top-K expert IDs and weights, looks up `layer.hot_id_map_host[expert]`, and writes compact hot/cold lists including counts. | Splits the GPU hot path from the CPU cold path per token/expert slot. |
| `llama_moe_hot_cache_build_worklist_from_logits(dst, logits, layer, ith, nth)` | For decode (`n_tokens == 1`), computes Top-K and softmax on CPU directly from logits and writes the same worklist. | Saves GPU TopK/weight nodes in the decode path and reduces routing overhead. |
| `llama_qwen35moe_hot_cache_build_worklist_op()` | `ggml_map_custom3` adapter for `llama_moe_hot_cache_build_worklist()`. | Binds our C++ logic as a GGML node in the graph. |
| `llama_qwen35moe_hot_cache_build_worklist_from_logits_op()` | `ggml_map_custom2` adapter for CPU routing directly from logits. | Enables the faster decode-routing path in the graph. |

Example:

```text
selected_experts = [7, 12, 44, 101]
hot cache contains 12 and 101

Hot list:
  cache_id(12), src_slot 1, weight ...
  cache_id(101), src_slot 3, weight ...

Cold list:
  expert 7, src_slot 0, weight ...
  expert 44, src_slot 2, weight ...
```

## Dynamic Update

File: `src/llama-moe-hot-cache.cpp`

| Method | What it does | How it helps |
| --- | --- | --- |
| `llama_moe_hot_cache_update_from_perf_json(model, json, update_rate)` | Reads current perf data, computes hit rate, finds desired experts per layer, creates evict/add pairs, sorts by gain, and exchanges at most `ceil(update_rate * hot_experts)` slots. | The cache adapts after each request without server restart and without full cache rebuild. |
| `copy_expert_slice()` in update context | Overwrites an existing cache slot with the new expert slice. | The VRAM buffer remains the same size; only contents and maps change. |
| `set_tensor_i32_1d()` / `set_tensor_f32_1d()` in update context | Updates `hot_id_map`, `hot_mask`, `cold_mask` for evict/add. | Graph routing sees the new cache layout from the next request. |

The update path does not change the number of hot experts per layer. It only
exchanges entries within already allocated slots. This keeps graph and buffer
shapes stable.

## Weighting

Files: `src/models/qwen35moe-hot-cache.cpp`, `src/models/gemma4-hot-cache.cpp`

| Method | What it does | How it helps |
| --- | --- | --- |
| `str_is()` / `str_is_any()` | String comparison for mode names. | Small local parser helpers without extra dependencies. |
| `normalize_layer_curve(value)` | Clamps to `0.0..1.0`, otherwise default `0.50`. | Prevents extreme or invalid weighting. |
| `layer_curve_from_env()` | Reads generic and old model-specific env names. | Preserves compatibility with existing start scripts. |
| `total_hits(layer, expert)` | Uses `hot+cold` or `raw`. | Weighting works with learning runs and hot-cache runs. |
| `score_to_u64(score)` | Robustly rounds weighted scores to `uint64_t`. | Ranking remains integer-based and sortable. |
| `layer_pressure(layer)` | Prefers join wait, otherwise lane delta, otherwise cold slots or wait-per-cold-slot. | Slow cold layers can receive more VRAM during cache selection. |
| `layer_pressure_for_source(layer, source)` | Selects between parallel pressure and total MoE time. | Supports `pressure` and `time` with the same infrastructure. |
| `average_layer_pressure()` | Average over layers with a valid pressure value. | Basis for relative pressure weighting. |
| `minmax_layer_pressure()` | Min/max over pressure values. | Basis for time and smooth normalization. |
| `percentile_sorted()` | Interpolated percentile from sorted values. | Helps robust smoothing without outlier dominance. |
| `robust_layer_pressure_bounds()` | Uses 10th and 90th percentile as pressure bounds. | `smooth` reacts less aggressively to individual outliers. |
| `pressure_stats()` | Builds average and bounds for one source. | Shared precomputation for all pressure strategies. |
| `pressure_weight()` | Scales layers relative to the average with bounded min/max. | Prefers layers that wait on cold work at the join. |
| `time_weight()` | Scales by normalized total MoE time. | Can prefer layers that are expensive overall. |
| `smooth_pressure_weight()` | Uses a square-root curve and bounded boost. | Smooths strong layer differences. |
| `sort_entries()` | Deterministic sorting by score, layer, expert. | Reproducible cache lists. |
| `score_with_layer_weight()` | Multiplies expert hits by layer weight and optionally adds a hot-sticky bonus. | Combines local expert popularity with global layer cost. |
| `qwen35moe_pressure_weighting::score()` | Scores with join/cold pressure. | Optimized for the observed CPU wait path. |
| `qwen35moe_smooth_pressure_weighting::score()` | Scores with robust smoothed pressure. | Experiment path for more even layer distribution. |
| `qwen35moe_time_weighting::score()` | Scores by total MoE time. | Alternative when join wait is not stable enough. |
| `qwen35moe_balanced_weighting::score()` | Ranks experts per layer and combines rank plus hits. | Forces stronger layer-wise distribution. |
| `qwen35moe_flat_weighting::score()` | Interleaved rank: first best expert of each layer, then second-best, etc. | Default because tests showed that more even layer occupancy is more stable under limited VRAM. |
| `weighting_strategy(mode)` | Returns a static strategy instance. | Avoids allocations and keeps mode selection centralized. |
| `llama_moe_hot_cache_weighting::parse_mode()` | Parses CLI/env modes including aliases. | Users can select `flat`, `pressure`, `smooth`, `time`, `balanced`. |
| `llama_moe_hot_cache_weighting::mode_name()` | Returns canonical name. | Logging and JSON remain readable. |
| `llama_moe_hot_cache_weighting::default_config()` | Builds default from env; default mode is `flat`. | Best-known default distribution without requiring CLI. |
| `llama_moe_hot_cache_weighting::score_observations()` | Public scoring entry. | Used by init and dynamic update. |
| `llama_moe_hot_cache_qwen35moe_weighting::*` | Thin wrapper around generic weighting. | Preserves old Qwen-specific API names. |
| `llama_moe_hot_cache_gemma4_weighting::score_observations()` | Uses generic weighting and allows Gemma4-specific layer-curve env. | Gemma4 gets the same strategies without touching Qwen code. |

## Graph Construction

File: `src/llama-moe-hot-cache-graph.cpp`

| Method | What it does | How it helps |
| --- | --- | --- |
| `llama_moe_hot_cache_graph_tweaks::parallel_mode()` | Reads `LLAMA_MOE_HOT_CACHE_PARALLEL`: `0/off`, `auto`, `force`. Default auto. | Hot/cold parallelization can be controlled without rebuild. |
| `parallel_min_slots()` | Reads minimum slots for the parallel region; default `2`. | Avoids parallel overhead for too-small workloads. |
| `merge_sum_rows()` | Enables merge through `ggml_sum_rows`. | Faster merge than many individual `ggml_add` nodes for multiple slots. |
| `cpu_decode_routing()` | Enables CPU worklist directly from logits in decode. | Saves routing nodes in the hot graph. |
| `decode_direct_merge()` | Enables direct decode merge to `[n_embd, 1]`. | Avoids materializing the full slot tensor. |
| `decode_strided_sum_rows()` | Allows sum_rows on non-contiguous strided layout. | Saves `ggml_cont` and copies in the merge. |
| `hot_dummy_padding()` | Reads dummy-padding switch. | Stable handling of unused hot slots. |
| `shared_input_row()` | Marks cold inputs as reusing the first row during decode. | Reduces gather cost in the cold path. |
| `cold_prefix_sum()` | Reduces only the valid cold-prefix slots. | Saves CPU work when few cold slots are active. |
| `cold_prefix_weighted_sum()` | Integrates weighting directly into prefix sum. | Saves a separate mul node in the cold decode path. |
| `decode_repeat_hot_input()` | Repeats hot input instead of gathering per slot. | Can reduce hot-gather overhead in decode. |
| `cold_first_row_input()` | Copies only the first input row for cold decode. | Special case for decode with shared input. |
| `branch_reduce_merge()` | Optional branch-internal reduce before join. | Experimental path, especially for Gemma4 comparison/fallback. |
| `env_enabled_by_default(name)` | Shared env parser for boolean tweaks. | All tweaks have a consistent default rule. |
| `llama_qwen35moe_hot_cache_graph_tweaks::get_profile()` | Qwen profile: direct decode optimizations, but `branch_reduce_merge` off. | Qwen stays on the proven fast path. |
| `llama_gemma4_hot_cache_graph_tweaks::get_profile()` | Gemma4 profile: direct decode optimizations and optional Branch-Reduce-Merge. | Gemma4 can use its own merge experiments without affecting Qwen. |
| `llama_moe_hot_cache_graph_profiles::profile_for_arch()` | Selects profile by architecture, empty by default. | New models must explicitly opt in; prevents side effects. |
| `llama_qwen35moe_hot_cache_sum_prefix_rows_op()` | Sums only the first `count` rows of a branch output. | Cold-prefix merge reduces CPU work for sparse cold lanes. |
| `llama_qwen35moe_hot_cache_sum_weighted_prefix_rows_op()` | Sums prefix rows with worklist weights. | Saves a previous weight-mul in the cold path. |
| `llama_qwen35moe_hot_cache_first_row_input_op()` | Copies the one decode input row into a branch tensor. | Helps the shared-input-row path. |
| `llama_moe_hot_cache_set_mul_mat_id_flags(t, flags)` | Writes flags into `op_params` of `ggml_mul_mat_id`. | Enables our kernel special cases without a new GGML op. |
| `llama_moe_hot_cache_build_lora_mm_id(graph, w, cur, ids, flags)` | Builds `ggml_mul_mat_id` plus LoRA parts with the same IDs/flags. | Hot/cold paths remain LoRA-compatible. |
| `llama_moe_hot_cache_build_moe_ffn_with_ids(...)` | Builds the actual expert FFN for a compact ID list: up/gate/gate-up, activation, down, scale tensors, weighting, output reduce. | Shared core for hot and cold branches, so both remain numerically equivalent to the normal MoE FFN. |
| `llama_moe_hot_cache_build_moe_hot_from_logits(graph, model, cur, logits, il, type_op)` | Generic hot-cache layer from precomputed router logits. Builds worklist, hot branch, cold branch, merge, and scheduler annotation. | Used by Gemma4 and reusable for additional models. |
| `llama_model_qwen35moe::graph::build_layer_ffn_hot(cur, il)` | Qwen-specific hot layer. Computes router logits itself, builds worklist, hot/cold branches, and merge. | Keeps the Qwen path stable and avoids changes to the standard Qwen FFN except the small gate hook. |
| `llama_model_gemma4::graph::build_layer_moe_hot(cur, logits, il)` | Gemma4 calls the generic hot path with `LLM_FFN_GELU`. | Gemma4 gets hot cache without a Qwen-specific copy. |

## Model Hooks

Files: `src/models/qwen35moe.cpp`, `src/models/gemma4.cpp`, `src/models/models.h`

| Hook | What it does | How it helps |
| --- | --- | --- |
| `llama_model_qwen35moe::graph::build_layer_ffn()` | Checks `llama_moe_hot_cache_layer_active(model, il)` and otherwise calls the normal `build_moe_ffn()`. | With hot cache disabled, Qwen stays on the upstream path. |
| `llama_model_gemma4::graph` MoE site | Computes Gemma router logits as before and calls `build_layer_moe_hot()` when the cache is active. | Gemma4 hot cache only touches the MoE branch and leaves attention/MLP unchanged. |
| Declarations in `models.h` | Adds `build_layer_ffn_hot()` and `build_layer_moe_hot()` to the model graph classes. | Minimal model-class interface for our separate hot-cache files. |

## Perf Collection And JSON

Files: `src/llama-moe-hot-cache-perf.cpp`, `src/llama-moe-hot-cache-perf.h`, `include/llama.h`

| Method | What it does | How it helps |
| --- | --- | --- |
| `llama_moe_layer_perf_local::ensure_shape_locked()` | Initializes layer and expert counter sizes. | Perf counters adapt to the currently loaded model. |
| `reset_locked(count_overflow)` | Resets all counters, timings, and debug fields. | Prevents stale data after mode changes or dynamic update. |
| `add_locked(dst, add)` | Saturating add with reset on overflow. | Long runs remain robust. |
| `add_expert_locked(layer, expert)` | Increments raw expert hit and total hits. | Learning runs without hot/cold split can create expert lists. |
| `add_branch_expert_locked(layer, expert, hot)` | Increments hot or cold expert counter. | UI and dynamic update see what was actually hot or cold in the last run. |
| `llama_moe_layer_perf_parse_mode(value, mode)` | Parses `full`, `update`, `off`. | Runtime switch through HTTP and env. |
| `llama_moe_layer_perf_mode_name(mode)` | Returns JSON/log name. | UI and logs show the active counting mode. |
| `llama_moe_layer_perf_env_mode()` | Reads `LLAMA_MOE_LAYER_PERF`, default `full`. | Start behavior can be controlled through env. |
| `llama_moe_layer_perf_get_mode(ctx)` | Combines runtime mode, `--no-perf`, and env. | `--no-perf` can truly disable counters. |
| `llama_moe_layer_perf_set_initial_mode(no_perf)` | Sets initial mode at server start. | UI dropdown correctly shows `off` after `--no-perf`. |
| `llama_moe_layer_perf_set_mode(mode)` | Changes mode and resets snapshot/counters. | HTTP POST can visibly reduce runtime cost. |
| `llama_moe_layer_perf_is_enabled(ctx)` | True except in `off`. | Central fast check. |
| `llama_moe_layer_perf_needs_expert_counts(no_perf)` | True for `full` and `update`. | Graph builds only needed counter nodes. |
| `llama_moe_parse_layer_from_name(name)` | Extracts layer ID from node names such as `ffn_moe_hot_down-12`. | Eval callback can associate nodes with layers. |
| `llama_moe_name_contains()` | String helper for node classification. | Small fast predicates instead of complex regex. |
| `llama_moe_is_*_node()` predicates | Detect TopK, hot/cold counts, gate/up/down, routing, merge, gather/scatter, and branch nodes. | Timings can be grouped by execution step. |
| `llama_moe_layer_perf_begin(n_layer, n_expert, n_expert_used)` | Activates the perf window for one graph compute. | Counters measure real inference only, not warmup. |
| `llama_moe_layer_perf_end()` | Deactivates the perf window. | Prevents accidental counting outside graph execution. |
| `llama_moe_layer_perf_reset()` | Public reset. | Server can count anew after dynamic update. |
| `llama_moe_layer_perf_has_data()` | Checks whether usable data exists. | Server does not write an empty perf file. |
| `llama_moe_layer_perf_count_topk_locked()` | Reads TopK IDs from backend tensor and counts raw experts. | Creates the initial expert list without hot cache. |
| `llama_moe_layer_perf_count_worklist_count_locked()` | Reads hot/cold count from worklist. | Computes hit rate and slots per layer. |
| `llama_moe_layer_perf_count_branch_experts_locked()` | Reads hot expert IDs or cold IDs and counts by branch. | Dynamic update knows which experts were active. |
| `llama_moe_layer_perf_eval_callback(t, ask, user_data)` | GGML eval callback: requests interesting nodes, reads IDs/counts after execution, and measures timings in full mode. | One callback site provides JSON, UI, and update data. |
| `llama_moe_layer_perf_collect_parallel_metrics(sched)` | Reads scheduler metrics for parallel regions and aggregates them per layer. | Shows join wait, overlap, fallbacks, and lane timings. |
| `llama_moe_layer_perf_graph_compute_begin(ctx, sched)` | Enables scheduler perf, sets eval callback, and starts perf window. | Integrates measurement into `llama_context::graph_compute()`. |
| `llama_moe_layer_perf_graph_compute_end(ctx, sched)` | Collects scheduler metrics, ends perf, and restores the original callback. | Prevents side effects on other llama.cpp callback users. |
| `llama_moe_layer_perf_json(ctx)` | Serializes summary and layer data; in `off` mode, minimal empty JSON. | API, UI, learning file, and dynamic update use the same format. |
| `llama_moe_layer_perf_json()` in `include/llama.h` | Public C API for JSON. | Server can access data without private headers. |

Perf modes:

| Mode | Contents | Performance impact |
| --- | --- | --- |
| `full` | Expert counters, hot/cold counters, all timings, fallback debug. | Highest overhead, best analysis. |
| `update` | Only counters needed by dynamic update, plus minimal parallel numbers. | Less overhead, update remains possible. |
| `off` | No counters, minimal JSON. | Largest speed gain. |

## Scheduler Parallelization

Files: `ggml/include/ggml-backend-moe-hot-cache.h`, `ggml/src/ggml-backend-moe-hot-cache.inc`, `ggml/src/ggml-backend.cpp`

| Method | What it does | How it helps |
| --- | --- | --- |
| `ggml_backend_sched_moe_hot_cache_parallel_init(sched)` | Allocates scheduler extension state. | Separates our data from normal scheduler code. |
| `ggml_backend_sched_moe_hot_cache_parallel_free(sched)` | Stops worker thread and deletes state. | No thread leaks on scheduler free. |
| `ggml_backend_sched_moe_hot_cache_parallel_reset(sched)` | Clears annotated regions before a new graph. | Old regions are not applied to new graphs. |
| `ggml_backend_sched_moe_hot_cache_parallel_is_split_boundary(sched, node)` | Forces split boundaries at hot start, cold start, and join. | Scheduler can recognize hot and cold lanes as separate split regions. |
| `ggml_backend_sched_moe_parallel_worker_loop(worker)` | Background thread computes cold or lane split ranges. | No thread start per token; a persistent worker lowers overhead. |
| `ggml_backend_sched_moe_parallel_worker_new()` | Creates worker and thread. | Lazy init: only when the parallel path is active. |
| `ggml_backend_sched_moe_parallel_worker_free(worker)` | Stops and joins worker. | Clean shutdown behavior. |
| `ggml_backend_sched_moe_parallel_worker_get(sched)` | Returns existing worker or creates one. | Reuse across many decode steps. |
| `ggml_backend_sched_moe_parallel_worker_start(...)` | Passes split range and compute state to worker. | Cold lane can run in parallel with hot lane. |
| `ggml_backend_sched_moe_parallel_worker_wait(...)` | Waits for worker completion and status. | Join point synchronizes correctly. |
| `ggml_backend_sched_find_split_containing(sched, tensor)` | Finds which scheduler split contains a tensor node. | Resolves annotated tensors to real split indexes. |
| `ggml_backend_sched_moe_parallel_split_backend_id()` | Returns backend ID of a split. | Debug and validation data. |
| `ggml_backend_sched_moe_parallel_fill_split_debug()` | Fills debug fields for splits and backends. | Explains fallbacks in `/moe-layer-perf`. |
| `ggml_backend_sched_backend_is_cuda()` | Checks backend name for CUDA. | Hot lane is parallelized only on CUDA. |
| `ggml_backend_sched_backend_is_cpu()` | Checks backend device type CPU. | Cold lane is accepted only as a CPU lane. |
| `ggml_backend_sched_moe_parallel_record_fallback()` | Writes fallback reason into perf structure. | UI/JSON show why parallelization did not apply. |
| `ggml_backend_sched_moe_parallel_fail_or_fallback()` | In `force`: error; in auto: log once and continue serially. | Safe experimentation without hard crashes in auto mode. |
| `ggml_backend_sched_resolve_moe_parallel_region()` | Validates split order, backends, count prefix, and join. | Prevents incorrect parallel execution for a different graph layout. |
| `ggml_backend_sched_read_f32_count()` | Reads hot/cold count from CPU or GPU. | Scheduler decides whether a lane is empty. |
| `ggml_backend_sched_moe_parallel_auto_min_slots()` | Reads minimum slots for auto parallel. | Small regions run serially because parallel overhead would be higher. |
| `ggml_backend_sched_zero_tensor(tensor)` | Zeroes skipped hot/cold output for an empty lane. | Merge remains numerically correct even when a lane has no work. |
| `ggml_backend_sched_compute_moe_parallel_region()` | Core: checks slot counts, starts cold worker, computes hot lane locally, synchronizes both, zeroes empty lanes, collects perf. | This is the actual hot/cold overlap. |
| `ggml_backend_sched_compute_splits(sched)` | Replaces normal split compute for annotated regions; serial otherwise. | Integration into the normal GGML scheduler with a minimal hook. |
| `ggml_backend_sched_set_moe_hot_cache_parallel_perf_enabled()` | Toggles scheduler perf measurement. | `--no-perf` and runtime mode reduce measurement overhead. |
| `ggml_backend_sched_moe_hot_cache_parallel_region()` | Public API for annotating a hot/cold/join region. | Graph code does not need to know scheduler internals. |
| `ggml_backend_sched_get_moe_hot_cache_parallel_perf()` | Copies scheduler perf data out. | Perf JSON can serialize lane timings and fallback reasons. |
| Hook in `ggml-backend.cpp` | Includes `.inc`, extends scheduler state, split boundaries, init/free/reset, and compute. | Keeps core diff as small as possible; logic lives in a separate `.inc` file. |

## GGML And CUDA Extensions

Files: `ggml/src/ggml-cuda/*`, `ggml/src/ggml-cpu/*`, `ggml/include/ggml.h`

| Area | What changed | How it helps |
| --- | --- | --- |
| `ggml_mul_mat_id` flags | `op_params` carry special cases for negative IDs, dummy IDs, and shared input row. | Hot/cold graph can cheaply skip invalid slots. |
| CUDA `mul_mat_id` | Interprets our flags for hot/cold slots. | GPU path processes compact hot IDs without extra reshaping. |
| CPU `mul_mat_id` | Supports the same flags in the cold path. | CPU cold lane remains numerically compatible and can skip invalid slots. |
| `ggml_cuda_sum_rows_utils::launch_f32_maybe_strided()` | Selects contiguous or strided `sum_rows` CUDA kernel. | Direct merge can avoid `ggml_cont` and sum faster. |
| `reduce_rows_f32_strided` sync adjustment | Adds PDL synchronization in the strided reduce path. | Stabilizes CUDA reduce for our strided merge. |
| `ggml_backend_sched_moe_hot_cache_parallel_is_split_boundary()` hook | Forces split boundaries at marked tensors. | Hot/cold parallel region can be mapped reliably to backend splits. |

These changes are the most conflict-prone locations during upstream rebases.
The actual feature logic therefore lives as much as possible in
`llama-moe-hot-cache-*` and `ggml-backend-moe-hot-cache.inc`.

## CLI And Common Parameters

Files: `common/arg.cpp`, `common/common.h`, `common/common.cpp`, `include/llama.h`

| Method/field | What it does | How it helps |
| --- | --- | --- |
| `common_params::moe_hot_cache_max_mib` | `0` off, positive MiB fixed, `-1` auto-sizing. | One argument covers static and automatic cache sizing. |
| `common_params::moe_hot_cache_auto_reserve_mib` | VRAM safety reserve during auto-sizing. | Prevents warmup or MTP OOM with tight memory. |
| `common_params::moe_hot_cache` | Path to perf JSON. | Decouples learning run and cache run. |
| `common_params::moe_hot_cache_update_rate` | Fraction of cache slots to exchange after request. | Dynamic cache without restart. |
| `common_params::moe_hot_cache_layer_curve` | Layer weighting strength. | User can choose smooth or aggressive distribution. |
| `common_params::moe_hot_cache_weighting` | Mode `flat`, `pressure`, `smooth`, `time`, `balanced`. | Experimenting without code changes. |
| `common_params::moe_layer_perf_out` | File for perf JSON after request. | First learning run does not need HTTP fetch. |
| `llama_moe_hot_cache_set_layer_curve_env()` | Mirrors CLI value into env. | Weighting code can read centrally from env/params. |
| `llama_moe_hot_cache_weighting_valid()` | Validates CLI mode. | Errors early at startup instead of later in cache init. |
| `llama_moe_hot_cache_set_weighting_env()` | Mirrors CLI mode into env. | Same behavior in old env path and new argument path. |
| Parse validations after `common_params_parse_ex()` | Require `--moe-hot-cache` when `max_mib != 0`, allow `-1` only with `--ctx-size`, require cache for update. | Misconfigurations fail before model startup. |
| `common_model_params_from_common_params()` extension | Copies hot-cache parameters into `llama_model_params`. | Loader and context have all required values without depending on global state. |
| Argument `--moe-hot-cache-max-mib` | Sets cache size. | Main feature activation switch. |
| Argument `--moe-hot-cache-auto-reserve-mib` | Sets safety reserve. | Tuning against OOM. |
| Argument `--moe-hot-cache` | Sets JSON file. | Selection source for hot experts. |
| Argument `--moe-hot-cache-update-rate` | Sets dynamic exchange rate. | Enables prompt-to-prompt adaptation. |
| Argument `--moe-hot-cache-layer-curve` | Sets layer curve. | Weighting can be changed per test. |
| Argument `--moe-hot-cache-weighting` | Sets weighting strategy. | Switch between flat, pressure, smooth, time, balanced. |
| Argument `--moe-layer-perf-out` | Writes JSON file and disables `no_perf`. | Learning run can directly create an expert list. |

## Server Integration

Files: `tools/server/server.cpp`, `tools/server/server-context.cpp`, `tools/server/server-context.h`, `tools/server/server-models.cpp`

| Method/hook | What it does | How it helps |
| --- | --- | --- |
| `llama_moe_layer_perf_set_initial_mode(params.no_perf)` during server start | Initializes perf mode according to `--no-perf`. | UI and backend start consistently. |
| Route `GET /moe-layer-perf` | Returns current or stored perf JSON. | UI and external tools can read live data. |
| Route `POST /moe-layer-perf` | Sets runtime mode `full/update/off`. | Counters can be reduced without server restart. |
| `remember_moe_layer_perf_json()` | Stores last snapshot. | After reset/update, the last usable view stays available. |
| `forget_moe_layer_perf_json()` | Deletes snapshot. | Mode switch does not show stale data. |
| `has_remembered_moe_layer_perf_json()` | Checks stored snapshot. | Helps with safe file writes. |
| `get_moe_layer_perf_json()` | Returns live data, snapshot, or empty JSON. | One source for route, file, and update. |
| `write_moe_layer_perf_file()` | Writes `--moe-layer-perf-out` when data exists. | Learning workflow automatically creates the JSON file. |
| `set_moe_layer_perf_mode()` | Calls perf-mode setter and deletes snapshot. | Runtime switch is applied centrally. |
| `update_moe_hot_cache_if_pending()` | After request: gets perf JSON, runs dynamic update, logs hit rate and exchange. | Server adapts cache only after a cleanly completed request. |
| `moe_hot_cache_update_pending` | Flag after request end. | Prevents updates in the middle of inference. |
| Router `server_models_routes::get_moe_layer_perf()` | Forwards `/moe-layer-perf` to the running model or requires `?model=` when ambiguous. | Router mode can address multiple model instances cleanly. |
| Router `server_models_routes::post_moe_layer_perf()` | Forwards mode changes to the target model. | UI dropdown also works in router mode. |

## UI

Files: `tools/ui/src/...`

| Method/component | What it does | How it helps |
| --- | --- | --- |
| `MoeLayerPerfService.get(model)` | Fetches `/moe-layer-perf`, in router mode with `model` and `autoload=false`. | UI can query the running model specifically. |
| `MoeLayerPerfService.setMode(mode, model)` | Sends `POST /moe-layer-perf`. | Runtime switch for `full/update/off`. |
| `modeFromResponse(data)` | Derives UI mode from JSON. | Dropdown stays in sync with backend. |
| `refreshMoePerfMode()` | Loads current mode for active model. | Initial dropdown state is correct after reload too. |
| `handleMoePerfModeChange(event)` | Optimistic mode change with rollback on error. | UI stays responsive; errors do not destroy the old state. |
| Button `ROUTES.MOE_LAYER_PERF` | Opens separate layer-perf page next to the prompt field. | Button remains reachable even when token/s display moves. |
| `clamp()` | Bounds values such as update interval. | UI accepts only 0.5 to 3 seconds. |
| `countsToMap()` | Converts JSON counts into maps. | Fast lookup per expert in the heatmap. |
| `addCounts()` | Adds count pairs into a map. | Hot/cold/raw counters can be combined. |
| `totalCountsForLayer()` | Builds total hits from hot+cold or raw fallback. | Active delta works with learning runs and hot-cache runs. |
| `activeDeltaForLayer()` | Compares current snapshot with previous snapshot. | Yellow cells show experts that were active since the last poll. |
| `toViewLayers()` | Sorts layers and adds maps. | Renders a stable UI structure. |
| `resolveExpertCount()` | Uses `n_expert` or highest expert ID. | Heatmap size works for reduced JSONs too. |
| `layerHitRate()` | Computes layer hit rate from ratio or slots. | Graph and cards show correct hit rates. |
| `resolveAverageHitRate()` | Uses summary ratio or layer average. | Top display remains available with different JSON modes. |
| `buildGraphPoints()` | Creates SVG points per layer hit rate. | Hit-rate progression across layers becomes visible. |
| `numberLocale()` / `formatDecimal()` / `formatInteger()` / `formatPercent()` | Localized number formatting. | Prevents confusing dot/comma display. |
| `metricValue()` | Reads summary fields type-safely. | Timings show `n/a` when a field is missing in update/off mode. |
| `formatMetricValue()` | Formats us and count values. | Timings remain compact and readable. |
| `formatMetricShare()` | Computes share of `total_moe_time_per_call_us`. | Shows where time goes in the MoE path. |
| `handleBack()` | Uses browser history, otherwise home page. | Back arrow returns to the previous chat. |
| `expertClass()` | Selects color: hot red, cold blue, active yellow, idle muted. | Fast visual check of cache occupancy. |
| `expertTitle()` | Tooltip per expert. | Debug without an additional detail page. |
| `expertLegendClass()` | Legend colors. | Consistency between grid and legend. |
| `handleIntervalInput()` | Normalizes polling interval. | Prevents too-fast or too-slow polling. |
| `refresh()` | Loads perf data, remembers previous snapshot and error state. | Live view updates without reset button. |
| `timingMetricCard()` | Renders a timing with tooltip and share. | Timing explanation directly in UI. |
| `timingMetricGroup()` | Groups timings by execution order. | Analysis follows routing, parallel region, lanes, sync, merge. |

## Tests

File: `tests/test-moe-hot-cache.cpp`

| Test/helper | What it checks | Why it matters |
| --- | --- | --- |
| `require_impl()` / `require` | Minimal test assert. | Test stays small without external framework. |
| `set_env_var()` / `scoped_env_var` | Temporary env changes. | Weighting defaults can be tested in isolation. |
| `test_default_weighting_is_flat()` | Default weighting is `flat`. | Protects the current default decision. |
| `test_parse_and_sort()` | JSON parsing and deterministic sorting. | Initial learning run is processed correctly. |
| `test_parse_branch_counts_and_layer_weight()` | `hot_experts`/`cold_experts` and pressure scoring. | Update data flows correctly into selection. |
| `test_qwen_layer_pressure_uses_total_wait()` | Qwen pressure prefers waiting layers. | Regression protection for layer curve. |
| `test_gemma4_layer_pressure_uses_total_wait()` | Gemma4 uses the same weighting infrastructure. | Protects Gemma4 from Qwen-special logic. |
| `test_flat_weighting_spreads_budget_over_layers()` | Flat spreads budget layer-wise. | Protects the default from regressing to pure hit sorting. |
| `test_parse_raw_opt_schema()` | New optimized JSON schema is read. | `/moe-layer-perf` and `--moe-layer-perf-out` stay compatible. |
| `test_select_budget()` | Budget including dummy expert. | Prevents VRAM overbooking. |
| `test_bad_schema()` | Wrong schema throws. | Early diagnosis for wrong file. |
| `make_ctx()` | Creates small GGML context. | Worklist tests run without a model. |
| `get_worklist_field()` | Reads one worklist field. | Tests check the exact packing layout. |
| `set_selected()` / `set_weight()` / `set_logit()` | Write test tensors. | Worklist tests remain understandable. |
| `require_close()` | Float comparison. | Softmax weights from logits are checked stably. |
| `test_build_worklist_mixed()` | Mixed hot/cold slots, padding, and counts. | Most important test for worklist correctness. |
| `test_build_worklist_all_hot_or_cold()` | Edge cases all hot or all cold. | Prevents invalid dummy and cold IDs. |
| `test_build_worklist_from_logits()` | CPU decode routing from logits including softmax and weight scale. | Protects the faster decode-routing path. |

## Documentation Commits

Commits `5a6edbb43`, `2b4944906`, and `27de9be40` contain no new runtime
methods, but document important decisions:

| File | Content | Benefit |
| --- | --- | --- |
| `README.md` | Quickstart, build, workflow, break-even graph, notes on `--cpu-moe`, auto-sizing, and UI. | User entry point. |
| `docs/development/moe-hot-cache-developer-guide.md` | German developer guide for architecture, workflow, tuning, and experiments. | Deeper technical context. |
| `docs/development/moe-hot-cache-developer-guide.en.md` | English version. | Shareable externally. |
| `docs/development/moe-hot-cache-runtime-switches.md` | Arguments and environment variables. | Reference for start scripts and INI configuration. |
| `docs/development/moe-hot-cache-parallelization-history.md` | History of performance levers. | Why specific paths default on/off. |
| `docs/development/moe-hot-cache-mtp-learnings*.md` | MTP results and why MTP was not a local default win. | Prevents repeating the same experiments. |
| `docs/development/moe-hot-cache-warm-lane-analysis.md` | Quadro M1200 / warm-lane analysis. | Documents why the second GPU does not pay off in this setup. |

## Rebase Risk

Low risk because logic lives in our own files:

- `src/llama-moe-hot-cache.cpp`
- `src/llama-moe-hot-cache-graph.cpp`
- `src/llama-moe-hot-cache-perf.cpp`
- `src/models/qwen35moe-hot-cache.cpp`
- `src/models/gemma4-hot-cache.cpp`
- `ggml/src/ggml-backend-moe-hot-cache.inc`
- UI page and documentation files

Higher risk during upstream rebase:

- `ggml/src/ggml-backend.cpp`: scheduler hooks and include of `.inc`.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: `mul_mat_id` flag semantics.
- `ggml/src/ggml-cpu/*`: CPU `mul_mat_id` flag semantics.
- `ggml/src/ggml-cuda/sumrows.cu` and `reduce_rows.cuh`: strided `sum_rows`.
- `src/models/qwen35moe.cpp` and `src/models/gemma4.cpp`: small hot-cache gates in the model graph.
- `common/arg.cpp`: CLI arguments near upstream argument blocks.

In the next refactor, these core hooks should remain as thin as possible: hook
in the upstream file, logic in our separate files.
