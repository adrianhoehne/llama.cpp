# MoE Hot-Cache and MTP: Learnings

Date: 2026-05-22

This note summarizes the experiments with Qwen3.6-35B-A3B MTP and the
MoE hot-cache path. It is intentionally implementation-oriented so the
experimental code can be reset while keeping enough detail to rebuild the useful
parts later.

## Short Conclusion

MTP did not provide a clear benefit for the local target setup. The hot-cache
path without MTP reached higher token rates than the MTP runs.

Observed direction:

- Hot-cache without MTP reached up to roughly `28 t/s` in earlier runs.
- One MTP run from the logs reached `25.33 t/s` eval rate, even with a high
  draft acceptance of about `94%`.
- Without hot-cache, but with coarse `override-tensor` layer placement, MTP
  often failed because of the additional MTP context memory.

The practical takeaway is: MTP is not free. It creates a second context against
the target model and needs additional CUDA compute / PP buffers. On a tight
12 GB VRAM setup, this extra memory pressure can cost more than the high draft
acceptance gives back.

## What MTP Does in llama.cpp

With `--spec-type draft-mtp`, llama.cpp does not load a completely separate
draft model when the target model contains MTP layers. Instead, after the normal
target context is created, llama.cpp creates a second MTP context against the
same target model:

```text
creating MTP draft context against the target model
llama_init_from_model(model_tgt, cparams_mtp)
```

This MTP context needs its own graph / compute memory. One failed run hit OOM
exactly at that point:

```text
allocating 800.02 MiB on device 0: cudaMalloc failed: out of memory
failed to create MTP context
```

Important: `--n-gpu-layers-draft auto` is of limited help for this MTP case,
because no separate draft model with its own layer offload is loaded. The
critical memory is the additional context / compute memory.

## Why MTP Failed to Load Without Hot-Cache

The non-hot-cache start command used coarse tensor overrides:

```text
blk.(0|1|2|3|4|5|6|40).ffn_...=CUDA0
```

This placed seven early MoE layers plus the full MTP layer 40 as complete expert
layers on CUDA0. That can still fit for normal inference, but it does not
necessarily leave enough reserve for the MTP context created afterwards.

Even though no `--ctx-size` appeared in the log, `--fit` selected a context size
automatically:

```text
n_ctx_seq (107008)
```

So the run effectively used an approximately 100k context, derived from the
memory fit. MTP then added more memory pressure on top.

## Why the MTP Layer Was Initially Cold

For Qwen3.6-35B-A3B, the MTP layer is typically layer `40`, because the model
has 40 normal transformer layers plus one NextN/MTP layer.

The hot-cache graph only uses the hot path when the layer was populated during
hot-cache initialization:

```cpp
llama_moe_hot_cache_layer_active(model, il)
```

If the startup JSON has no data for layer 40, the normal hot-cache selection
cannot choose experts for layer 40. This caused the live snapshot to show:

```text
layer = 40
hot_slots_total = 0
cold_slots_total = 76384
hot_experts_count = 0
cold_experts_count = 254
```

The auto-update path could not fix this state because it only updates already
active hot-cache layers. A fully inactive layer has no hot expert slots that can
be replaced.

## Rejected Approach: Random Fallback

One temporary approach was to fill layers without perf data with random experts.
That was removed again.

Reasons not to keep it:

- Random selection makes results hard to reproduce.
- It hides whether the perf JSON actually contains layer data.
- For rebase and review, random fallback is harder to justify.

The better approach is an explicit MTP priority path that only applies to
recognized MTP layers and behaves deterministically.

## Better Approach: Explicit MTP Layer Priority

The tested control was:

```text
--moe-hot-cache-mtp-layer-ratio N
```

Semantics:

- `0.0` = no MTP special handling
- `0.9` = prioritize about 90% of the MTP experts
- `1.0` = prioritize the whole MTP layer
- unset = auto
  - with `draft-mtp`: `1.0`
  - without `draft-mtp`: `0.0`

For `n_expert = 256`:

```text
ratio 0.9 -> ceil(256 * 0.9) = 231 experts
ratio 1.0 -> 256 experts
```

A ratio below `1.0` can be useful when some MTP experts are effectively dead.
That leaves hot-cache slots for other layers.

## Reimplementation: Required Parameters

Use an auto sentinel in `common_params`:

```cpp
float moe_hot_cache_mtp_layer_ratio = -1.0f; // -1 = auto
```

Add CLI / INI support:

```text
--moe-hot-cache-mtp-layer-ratio N
LLAMA_ARG_MOE_HOT_CACHE_MTP_LAYER_RATIO=N
```

Validation:

```cpp
if (value < 0.0f || value > 1.0f) {
    throw std::invalid_argument("--moe-hot-cache-mtp-layer-ratio must be between 0.0 and 1.0");
}
```

When converting from `common_params` to `llama_model_params`:

```cpp
const bool mtp_enabled =
    std::find(params.speculative.types.begin(),
              params.speculative.types.end(),
              COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

mparams.moe_hot_cache_mtp_layer_ratio =
    params.moe_hot_cache_mtp_layer_ratio >= 0.0f
        ? params.moe_hot_cache_mtp_layer_ratio
        : (mtp_enabled ? 1.0f : 0.0f);
```

Add the field to `llama_model_params`:

```cpp
float moe_hot_cache_mtp_layer_ratio;
```

Default in `llama_model_default_params()`:

```cpp
/*.moe_hot_cache_mtp_layer_ratio =*/ 0.0f,
```

## Reimplementation: Detecting MTP Layers

For Qwen35MoE / Qwen3.6-MoE, the MTP layer can be detected via
`nextn_predict_layers`.

Pseudocode:

```cpp
static std::vector<uint32_t> mtp_hot_cache_layers(const llama_model & model) {
    std::vector<uint32_t> layers;

    if (model.arch != LLM_ARCH_QWEN35MOE ||
        model.hparams.nextn_predict_layers == 0 ||
        model.hparams.nextn_predict_layers >= model.hparams.n_layer) {
        return layers;
    }

    const uint32_t first_mtp_layer =
        model.hparams.n_layer - model.hparams.nextn_predict_layers;

    for (uint32_t il = first_mtp_layer; il < model.hparams.n_layer; ++il) {
        if (model.layers[il].ffn_down_exps != nullptr) {
            layers.push_back(il);
        }
    }

    return layers;
}
```

This is intentionally narrower than a general fallback. Only real MTP MoE
layers are prioritized.

## Reimplementation: Selection With MTP Priority

During initial hot-cache construction, build a priority list before the normal
budget selection:

```cpp
struct priority_layer {
    uint32_t layer;
    size_t target_experts;
    size_t total_experts;
    float ratio;
};
```

Target calculation:

```cpp
target = min(total, ceil(total * ratio));
```

Selection strategy:

1. Take observed experts from the prioritized layer first, sorted by the normal
   score from the perf JSON.
2. If fewer than `target` are available, deterministically fill from the real
   model expert sizes.
3. Append non-prioritized layers using the normal ranking.
4. Reuse the existing budget selector.

Pseudocode:

```cpp
for entry in observed:
    if entry.layer is priority and selected[layer] < target[layer]:
        add(entry, max_score)

for size in expert_sizes:
    if size.layer is priority and selected[layer] < target[layer]:
        add({ size.layer, size.expert, max_score })

for entry in observed:
    if entry.layer is not priority:
        add(entry)

plan = select_by_budget(prioritized, expert_sizes, budget_bytes)
```

Important: deterministic filling is not a global fallback. It only applies to
explicitly prioritized MTP layers.

Expected log line:

```text
MTP hot-cache priority: layer 40 selected 231/256 experts (target = 231, ratio = 90.00%)
```

## Reimplementation: Connecting the MTP Graph to Hot-Cache

The MTP graph needs a narrow hook for the MoE FFN. To reduce upstream rebase
conflicts, keep as little code as possible in `qwen35moe.cpp`.

Minimal change in `qwen35moe.cpp`:

```cpp
ggml_tensor * moe_out = build_layer_ffn(cur, il);
cb(moe_out, "mtp_ffn_moe_out", il);
```

The actual implementation can live in `llama-moe-hot-cache-graph.cpp`:

```cpp
ggml_tensor * llama_model_qwen35moe::graph_mtp::build_layer_ffn(
        ggml_tensor * cur,
        const int il) {
    const auto & layer = model.layers[il];

    ggml_tensor * logits = build_lora_mm(layer.ffn_gate_inp, cur);
    cb(logits, "ffn_moe_logits", il);

    if (llama_moe_hot_cache_layer_active(model, il)) {
        return llama_moe_hot_cache_build_moe_hot_from_logits(
            *this, model, cur, logits, il, LLM_FFN_SILU);
    }

    return build_moe_ffn(cur,
        nullptr,
        layer.ffn_up_exps,
        layer.ffn_gate_exps,
        layer.ffn_down_exps,
        nullptr,
        n_expert, n_expert_used,
        LLM_FFN_SILU, true,
        hparams.expert_weights_scale,
        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
        logits, layer.ffn_gate_up_exps,
        layer.ffn_up_exps_s,
        layer.ffn_gate_exps_s,
        layer.ffn_down_exps_s);
}
```

Why explicitly build `logits`:

- The hot-cache path needs the router logits to build hot/cold worklists from
  the same gate results.
- The fallback path can pass the same logits to `build_moe_ffn` instead of
  recomputing gate logits.

## Reimplementation: MTP Auto-Update

If the MTP layer is active at startup, the dynamic update can update it like
other layers. This is especially useful with a ratio below `1.0`:

- Start with `0.9`, for example 231 hot experts.
- After the prompt, dead experts inside those 231 slots can be replaced with
  better experts.
- The layer size remains fixed; the update does not resize the layer.

The update path should collect MTP-specific stats:

```cpp
bool mtp_layer_active;
bool mtp_layer_full;
uint32_t mtp_layer;
size_t mtp_hot_experts;
size_t mtp_total_experts;
size_t mtp_candidates;
size_t mtp_exchanged;
```

Server log after update:

```text
MoE hot-cache MTP layer update:
layer = 40, hot experts = 231/256, candidates = ..., exchanged = ...
```

Limitation: a completely inactive MTP layer cannot be activated by auto-update
after startup. It needs at least one hot-cache entry at initialization time.

## Memory Learnings

MTP needs additional reserve. For hot-cache auto-sizing, use a more conservative
reserve when MTP is enabled:

```text
--moe-hot-cache-max-mib -1
--moe-hot-cache-auto-reserve-mib 1600
```

The exact value is hardware- and context-dependent. In the logs, `800 MiB` for
the MTP context compute buffer was enough to trigger OOM. `1024 MiB` reserve can
be tight when warmup, graph reserve, and CUDA transients are added.

Without hot-cache, but with `override-tensor`, remember:

- complete MoE layers on CUDA0 are very coarse,
- a complete MTP layer 40 costs much more than a selected expert subset,
- normal inference can fit while MTP still fails afterwards.

## Performance Learnings

High draft acceptance does not guarantee a speedup. In the observed MTP run:

```text
draft acceptance = 0.94107
eval time = 185305.21 ms / 4694 tokens = 25.33 t/s
```

Despite about 94% acceptance, the end rate was lower than good hot-cache runs
without MTP. Likely causes:

- the additional MTP context increases memory and graph pressure,
- the MTP layer itself is another MoE layer,
- tight VRAM forces a smaller hot-cache,
- less hot-cache budget can increase cold work in other layers,
- the CPU/GPU balance of the hot/cold path was already better without MTP.

Practical conclusion for this setup:

- MTP is not the best default for Qwen3.6-35B-A3B on the local RTX 2060 setup.
- Hot-cache without MTP is the better performance path.
- MTP can be retested later with more VRAM or if llama.cpp significantly
  reduces the MTP context memory.

## Rebase and Code Isolation Learnings

For a rebase-friendly reimplementation:

- Do not touch `common/speculative.cpp`. The local workaround was removed
  because upstream is likely to fix that area independently.
- Do not leave broad logic in `qwen35moe.cpp`.
- Keep `qwen35moe.cpp` to a narrow hook:
  - `graph_mtp` stores `model`,
  - MTP MoE calls `build_layer_ffn(cur, il)`.
- Put the real logic in hot-cache files:
  - `llama-moe-hot-cache.cpp` for selection, budget, and update,
  - `llama-moe-hot-cache-graph.cpp` for graph helpers.
- Do not use random fallback.
- Bind MTP priority strictly to detected MTP layers.

Not easily avoidable if MTP hot-cache is integrated into the graph:

- A small declaration in `models.h`.
- A small call in `qwen35moe.cpp`.

## Suggested Reset Scope

If the experimental MTP code is discarded, the tracked changes in these files
can be reset:

- `common/arg.cpp`
- `common/common.cpp`
- `common/common.h`
- `include/llama.h`
- `src/llama-model.cpp`
- `src/llama-moe-hot-cache-graph.cpp`
- `src/llama-moe-hot-cache.cpp`
- `src/llama-moe-hot-cache.h`
- `src/models/models.h`
- `src/models/qwen35moe.cpp`
- `tools/server/server-context.cpp`

Check untracked local test files separately. Do not blindly delete them if they
still contain perf data, launch scripts, or model configuration.

## If MTP Is Tested Again Later

Recommended test plan:

1. Measure baseline without MTP.
2. Test MTP without hot-cache only when enough VRAM is free.
3. Test MTP with hot-cache and conservative reserve:

```text
--moe-hot-cache-max-mib -1
--moe-hot-cache-auto-reserve-mib 1600
--moe-hot-cache-mtp-layer-ratio 0.9
```

4. After the first long prompt, inspect:

```text
/moe-layer-perf
```

Most relevant fields:

- layer 40 `hot_slots_total`
- layer 40 `cold_slots_total`
- layer 40 `hot_experts_count`
- global hot slot ratio
- end-to-end `eval t/s`
- MTP `draft acceptance`

Stop criterion:

If MTP delivers lower `t/s` than hot-cache without MTP despite high acceptance,
keep MTP disabled for this hardware / context setup.
