#include "llama-moe-hot-cache-builder.h"

#include "llama-moe-hot-cache-planner.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace {

static ggml_tensor * new_tensor_like_experts(
        ggml_context * ctx,
        const ggml_tensor * src,
        int64_t n_cache,
        const char * name) {
    if (src == nullptr) {
        return nullptr;
    }

    ggml_tensor * dst = ggml_new_tensor_3d(ctx, src->type, src->ne[0], src->ne[1], n_cache);
    ggml_set_name(dst, name);
    return dst;
}

static ggml_tensor * new_tensor_like_scale(
        ggml_context * ctx,
        const ggml_tensor * src,
        int64_t n_cache,
        const char * name) {
    if (src == nullptr) {
        return nullptr;
    }

    ggml_tensor * dst = ggml_new_tensor_1d(ctx, src->type, n_cache);
    ggml_set_name(dst, name);
    return dst;
}

static void zero_tensor(ggml_tensor * t) {
    std::vector<uint8_t> zeros(ggml_nbytes(t), 0);
    ggml_backend_tensor_set(t, zeros.data(), 0, zeros.size());
}

} // namespace

std::vector<std::vector<uint32_t>> llama_moe_hot_cache_group_selected_by_layer(
        const llama_moe_hot_cache_plan & plan,
        size_t n_layer) {
    std::vector<std::vector<uint32_t>> selected_by_layer(n_layer);
    for (const auto & selected : plan.selected) {
        if (selected.layer < selected_by_layer.size()) {
            selected_by_layer[selected.layer].push_back(selected.expert);
        }
    }
    return selected_by_layer;
}

llama_moe_hot_cache_layer_selection_stats llama_moe_hot_cache_summarize_selected_layers(
        const std::vector<std::vector<uint32_t>> & selected_by_layer) {
    llama_moe_hot_cache_layer_selection_stats stats;
    stats.min_hot = std::numeric_limits<size_t>::max();

    for (const auto & experts : selected_by_layer) {
        if (experts.empty()) {
            continue;
        }

        stats.active_layers++;
        stats.total_hot += experts.size();
        stats.min_hot = std::min(stats.min_hot, experts.size());
        stats.max_hot = std::max(stats.max_hot, experts.size());
    }

    if (stats.active_layers == 0) {
        stats.min_hot = 0;
    }

    return stats;
}

void llama_moe_hot_cache_copy_expert_slice(
        const ggml_tensor * src,
        ggml_tensor * dst,
        uint32_t src_expert,
        uint32_t dst_expert) {
    if (src == nullptr || dst == nullptr) {
        return;
    }

    const size_t bytes = llama_moe_hot_cache_tensor_expert_bytes(src);
    std::vector<uint8_t> buf(bytes);
    ggml_backend_tensor_get(src, buf.data(), src->nb[2]*src_expert, bytes);
    ggml_backend_tensor_set(dst, buf.data(), dst->nb[2]*dst_expert, bytes);
}

void llama_moe_hot_cache_copy_scale_slice(
        const ggml_tensor * src,
        ggml_tensor * dst,
        uint32_t src_expert,
        uint32_t dst_expert) {
    if (src == nullptr || dst == nullptr) {
        return;
    }

    const size_t bytes = ggml_nbytes(src) / size_t(src->ne[0]);
    std::vector<uint8_t> buf(bytes);
    ggml_backend_tensor_get(src, buf.data(), src->nb[0]*src_expert, bytes);
    ggml_backend_tensor_set(dst, buf.data(), dst->nb[0]*dst_expert, bytes);
}

void llama_moe_hot_cache_set_tensor_i32_1d(ggml_tensor * t, uint32_t index, int32_t value) {
    ggml_backend_tensor_set(t, &value, t->nb[1]*index, sizeof(value));
}

void llama_moe_hot_cache_set_tensor_f32_1d(ggml_tensor * t, uint32_t index, float value) {
    ggml_backend_tensor_set(t, &value, t->nb[1]*index, sizeof(value));
}

std::unique_ptr<llama_moe_hot_cache> llama_moe_hot_cache_build(
        const llama_model & model,
        const llama_moe_hot_cache_plan & plan,
        ggml_backend_dev_t cache_dev) {
    auto cache = std::make_unique<llama_moe_hot_cache>();
    cache->layers.resize(model.hparams.n_layer);

    const auto selected_by_layer = llama_moe_hot_cache_group_selected_by_layer(plan, model.hparams.n_layer);
    const auto stats = llama_moe_hot_cache_summarize_selected_layers(selected_by_layer);

    LLAMA_LOG_INFO("%s: hot-cache active layers = %zu/%zu, hot experts per active layer min/avg/max = %zu/%.1f/%zu\n",
            __func__,
            stats.active_layers,
            selected_by_layer.size(),
            stats.min_hot,
            stats.avg_hot(),
            stats.max_hot);

    size_t n_tensors = 0;
    for (uint32_t il = 0; il < selected_by_layer.size(); ++il) {
        if (selected_by_layer[il].empty()) {
            continue;
        }

        const auto & layer = model.layers[il];

        n_tensors += 4; // map + hot mask + cold mask + down
        n_tensors += layer.ffn_gate_up_exps != nullptr ? 1 : 2;
        n_tensors += layer.ffn_gate_exps_s != nullptr ? 1 : 0;
        n_tensors += layer.ffn_up_exps_s   != nullptr ? 1 : 0;
        n_tensors += layer.ffn_down_exps_s != nullptr ? 1 : 0;

        cache->layers[il].n_hot = selected_by_layer[il].size();
        cache->layers[il].n_expert = layer.ffn_down_exps ? layer.ffn_down_exps->ne[2] : 0;
        cache->layers[il].expert_weights_scale = model.hparams.expert_weights_scale;
    }

    // Reserve extra space per tensor for internal ggml bookkeeping
    // (hash map entries, linked-list pointers, etc. beyond ggml_tensor_overhead()).
    static constexpr size_t EXTRA_PER_TENSOR = 64;
    const size_t n_extra_tensors = std::max<size_t>(64, n_tensors / 4);
    const size_t ctx_mem_size = ggml_tensor_overhead() * (n_tensors + n_extra_tensors)
                              + EXTRA_PER_TENSOR * n_tensors;

    ggml_init_params ctx_params = {
        /*.mem_size   =*/ ctx_mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx { ggml_init(ctx_params) };
    if (!ctx) {
        throw std::runtime_error("failed to create MoE hot-cache ggml context");
    }

    for (uint32_t il = 0; il < selected_by_layer.size(); ++il) {
        const auto & experts = selected_by_layer[il];
        if (experts.empty()) {
            continue;
        }

        const auto & src = model.layers[il];
        auto & dst = cache->layers[il];
        const int64_t n_cache = int64_t(experts.size()) + 1;
        const int64_t n_expert = src.ffn_down_exps->ne[2];

        dst.ffn_gate_up_exps = new_tensor_like_experts(ctx.get(), src.ffn_gate_up_exps, n_cache, format("blk.%u.ffn_gate_up_exps.hot_cache", il).c_str());
        dst.ffn_gate_exps    = new_tensor_like_experts(ctx.get(), src.ffn_gate_exps,    n_cache, format("blk.%u.ffn_gate_exps.hot_cache",    il).c_str());
        dst.ffn_up_exps      = new_tensor_like_experts(ctx.get(), src.ffn_up_exps,      n_cache, format("blk.%u.ffn_up_exps.hot_cache",      il).c_str());
        dst.ffn_down_exps    = new_tensor_like_experts(ctx.get(), src.ffn_down_exps,    n_cache, format("blk.%u.ffn_down_exps.hot_cache",    il).c_str());
        dst.ffn_gate_exps_s  = new_tensor_like_scale  (ctx.get(), src.ffn_gate_exps_s,  n_cache, format("blk.%u.ffn_gate_exps_s.hot_cache",  il).c_str());
        dst.ffn_up_exps_s    = new_tensor_like_scale  (ctx.get(), src.ffn_up_exps_s,    n_cache, format("blk.%u.ffn_up_exps_s.hot_cache",    il).c_str());
        dst.ffn_down_exps_s  = new_tensor_like_scale  (ctx.get(), src.ffn_down_exps_s,  n_cache, format("blk.%u.ffn_down_exps_s.hot_cache",  il).c_str());

        dst.hot_id_map = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, n_expert);
        ggml_format_name(dst.hot_id_map, "blk.%u.moe_hot_id_map", il);
        dst.hot_mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, n_expert);
        ggml_format_name(dst.hot_mask, "blk.%u.moe_hot_mask", il);
        dst.cold_mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, n_expert);
        ggml_format_name(dst.cold_mask, "blk.%u.moe_cold_mask", il);
    }

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(cache_dev);
    ggml_backend_buffer_ptr buf { ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft) };
    if (!buf) {
        throw std::runtime_error("failed to allocate MoE hot-cache buffer");
    }
    ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    for (uint32_t il = 0; il < selected_by_layer.size(); ++il) {
        const auto & experts = selected_by_layer[il];
        if (experts.empty()) {
            continue;
        }

        const auto & src = model.layers[il];
        auto & dst = cache->layers[il];
        const uint32_t dummy_id = experts.size();
        const uint32_t n_expert = dst.n_expert;

        if (dst.ffn_gate_up_exps) { zero_tensor(dst.ffn_gate_up_exps); }
        if (dst.ffn_gate_exps)    { zero_tensor(dst.ffn_gate_exps); }
        if (dst.ffn_up_exps)      { zero_tensor(dst.ffn_up_exps); }
        if (dst.ffn_down_exps)    { zero_tensor(dst.ffn_down_exps); }
        if (dst.ffn_gate_exps_s)  { zero_tensor(dst.ffn_gate_exps_s); }
        if (dst.ffn_up_exps_s)    { zero_tensor(dst.ffn_up_exps_s); }
        if (dst.ffn_down_exps_s)  { zero_tensor(dst.ffn_down_exps_s); }

        std::vector<int32_t> hot_id_map(n_expert, int32_t(dummy_id));
        std::vector<float> hot_mask(n_expert, 0.0f);
        std::vector<float> cold_mask(n_expert, 1.0f);
        dst.hot_id_map_host.assign(n_expert, -1);

        for (uint32_t cache_id = 0; cache_id < experts.size(); ++cache_id) {
            const uint32_t expert = experts[cache_id];
            if (expert >= n_expert) {
                continue;
            }

            hot_id_map[expert] = int32_t(cache_id);
            hot_mask[expert] = 1.0f;
            cold_mask[expert] = 0.0f;
            dst.hot_id_map_host[expert] = int32_t(cache_id);

            llama_moe_hot_cache_copy_expert_slice(src.ffn_gate_up_exps, dst.ffn_gate_up_exps, expert, cache_id);
            llama_moe_hot_cache_copy_expert_slice(src.ffn_gate_exps,    dst.ffn_gate_exps,    expert, cache_id);
            llama_moe_hot_cache_copy_expert_slice(src.ffn_up_exps,      dst.ffn_up_exps,      expert, cache_id);
            llama_moe_hot_cache_copy_expert_slice(src.ffn_down_exps,    dst.ffn_down_exps,    expert, cache_id);
            llama_moe_hot_cache_copy_scale_slice(src.ffn_gate_exps_s,   dst.ffn_gate_exps_s,  expert, cache_id);
            llama_moe_hot_cache_copy_scale_slice(src.ffn_up_exps_s,     dst.ffn_up_exps_s,    expert, cache_id);
            llama_moe_hot_cache_copy_scale_slice(src.ffn_down_exps_s,   dst.ffn_down_exps_s,  expert, cache_id);
        }

        ggml_backend_tensor_set(dst.hot_id_map, hot_id_map.data(), 0, hot_id_map.size()*sizeof(hot_id_map[0]));
        ggml_backend_tensor_set(dst.hot_mask,   hot_mask.data(),   0, hot_mask.size()*sizeof(hot_mask[0]));
        ggml_backend_tensor_set(dst.cold_mask,  cold_mask.data(),  0, cold_mask.size()*sizeof(cold_mask[0]));
    }

    LLAMA_LOG_WARN("%s: %12s hot-cache buffer size = %8.2f MiB\n",
            __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get())/1024.0/1024.0);

    cache->bufs.emplace_back(std::move(buf));
    cache->ctxs.emplace_back(std::move(ctx));
    return cache;
}
