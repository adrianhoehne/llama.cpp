#include "llama-moe-hot-cache.h"
#include "llama-moe-hot-cache-budget.h"
#include "llama-moe-hot-cache-common.h"
#include "llama-moe-hot-cache-parser.h"
#include "llama-moe-hot-cache-planner.h"

#include "llama-model.h"
#include "llama-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {

static bool llama_moe_hot_cache_hot_dummy_padding() {
    static const bool enabled = []() {
        const char * env = std::getenv("LLAMA_MOE_HOT_CACHE_HOT_DUMMY_PADDING");
        return env == nullptr || env[0] == '\0' ||
               (std::strcmp(env, "0") != 0 && std::strcmp(env, "off") != 0 && std::strcmp(env, "false") != 0);
    }();

    return enabled;
}

static void add_saturating(uint64_t & dst, uint64_t value) {
    if (dst > std::numeric_limits<uint64_t>::max() - value) {
        dst = std::numeric_limits<uint64_t>::max();
    } else {
        dst += value;
    }
}

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

static void zero_tensor(ggml_tensor * t) {
    std::vector<uint8_t> zeros(ggml_nbytes(t), 0);
    ggml_backend_tensor_set(t, zeros.data(), 0, zeros.size());
}

static void copy_expert_slice(const ggml_tensor * src, ggml_tensor * dst, uint32_t src_expert, uint32_t dst_expert) {
    if (src == nullptr || dst == nullptr) {
        return;
    }

    const size_t bytes = llama_moe_hot_cache_tensor_expert_bytes(src);
    std::vector<uint8_t> buf(bytes);
    ggml_backend_tensor_get(src, buf.data(), src->nb[2]*src_expert, bytes);
    ggml_backend_tensor_set(dst, buf.data(), dst->nb[2]*dst_expert, bytes);
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

static void copy_scale_slice(const ggml_tensor * src, ggml_tensor * dst, uint32_t src_expert, uint32_t dst_expert) {
    if (src == nullptr || dst == nullptr) {
        return;
    }

    const size_t bytes = ggml_nbytes(src) / size_t(src->ne[0]);
    std::vector<uint8_t> buf(bytes);
    ggml_backend_tensor_get(src, buf.data(), src->nb[0]*src_expert, bytes);
    ggml_backend_tensor_set(dst, buf.data(), dst->nb[0]*dst_expert, bytes);
}

static void set_tensor_i32_1d(ggml_tensor * t, uint32_t index, int32_t value) {
    ggml_backend_tensor_set(t, &value, t->nb[1]*index, sizeof(value));
}

static void set_tensor_f32_1d(ggml_tensor * t, uint32_t index, float value) {
    ggml_backend_tensor_set(t, &value, t->nb[1]*index, sizeof(value));
}

static std::vector<uint32_t> current_hot_experts(const llama_moe_hot_cache_layer & layer) {
    std::vector<uint32_t> result(layer.n_hot, std::numeric_limits<uint32_t>::max());

    for (uint32_t expert = 0; expert < layer.hot_id_map_host.size(); ++expert) {
        const int32_t cache_id = layer.hot_id_map_host[expert];
        if (cache_id >= 0 && uint32_t(cache_id) < result.size()) {
            result[cache_id] = expert;
        }
    }

    result.erase(std::remove(result.begin(), result.end(), std::numeric_limits<uint32_t>::max()), result.end());
    return result;
}

static uint64_t observation_total_hits(
        const llama_moe_hot_cache_layer_observation & layer,
        const llama_moe_hot_cache_expert_observation & expert) {
    uint64_t total = expert.raw;
    if (layer.has_branch_counts) {
        total = expert.hot;
        add_saturating(total, expert.cold);
    }

    return total;
}

static void sort_hot_cache_entries(std::vector<llama_moe_hot_cache_entry> & entries) {
    std::sort(entries.begin(), entries.end(), [](const auto & a, const auto & b) {
        if (a.hit_count != b.hit_count) {
            return a.hit_count > b.hit_count;
        }
        if (a.layer != b.layer) {
            return a.layer < b.layer;
        }
        return a.expert < b.expert;
    });
}

static std::vector<llama_moe_hot_cache_entry> score_observations_default(
        const std::vector<llama_moe_hot_cache_layer_observation> & observations) {
    std::vector<llama_moe_hot_cache_entry> result;
    for (const auto & obs : observations) {
        for (const auto & expert : obs.experts) {
            const uint64_t total_hits = observation_total_hits(obs, expert);
            if (total_hits == 0) {
                continue;
            }

            result.push_back({ obs.layer, expert.expert, total_hits });
        }
    }

    sort_hot_cache_entries(result);
    return result;
}

static llama_moe_hot_cache_weighting_config weighting_config_from_params(
        const llama_model_params * params) {
    auto config = llama_moe_hot_cache_weighting::default_config();
    if (params == nullptr) {
        return config;
    }

    const char * curve_env = std::getenv("LLAMA_MOE_HOT_CACHE_LAYER_CURVE");
    if (curve_env == nullptr || curve_env[0] == '\0') {
        curve_env = std::getenv("LLAMA_MOE_HOT_CACHE_QWEN_LAYER_CURVE");
    }
    if (curve_env == nullptr || curve_env[0] == '\0') {
        curve_env = std::getenv("LLAMA_MOE_HOT_CACHE_GEMMA4_LAYER_CURVE");
    }
    if (curve_env == nullptr || curve_env[0] == '\0') {
        config.layer_curve = params->moe_hot_cache_layer_curve;
    }
    if (params->moe_hot_cache_weighting != nullptr && params->moe_hot_cache_weighting[0] != '\0' &&
        !llama_moe_hot_cache_weighting::parse_mode(params->moe_hot_cache_weighting, config.mode)) {
        LLAMA_LOG_WARN("%s: unknown MoE hot-cache weighting '%s', using %s\n",
                __func__, params->moe_hot_cache_weighting,
                llama_moe_hot_cache_weighting::mode_name(config.mode));
    }

    return config;
}

static std::vector<llama_moe_hot_cache_entry> score_observations_for_arch(
        llm_arch arch,
        const std::vector<llama_moe_hot_cache_layer_observation> & observations,
        const llama_model_params * params = nullptr) {
    switch (arch) {
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_GEMMA4:
        default: {
            const auto config = weighting_config_from_params(params);
            return llama_moe_hot_cache_weighting::score_observations(observations, config);
        }
    }
}

} // namespace

std::vector<llama_moe_hot_cache_layer_observation> llama_moe_hot_cache_parse_perf_json_observations(
        const std::string & json_str) {
    return llama_moe_hot_cache_perf_json_parser::parse_observations(json_str);
}

std::vector<llama_moe_hot_cache_entry> llama_moe_hot_cache_parse_perf_json(const std::string & json_str) {
    const auto observations = llama_moe_hot_cache_parse_perf_json_observations(json_str);
    return score_observations_default(observations);
}

void llama_moe_hot_cache_init(llama_model & model, const llama_model_params & params, bool reserve_kv_cache) {
    if (params.moe_hot_cache_max_mib == 0) {
        return;
    }

    if (model.moe_hot_cache != nullptr) {
        return;
    }

    if (params.no_alloc) {
        LLAMA_LOG_INFO("%s: skipping hot-cache build during no-alloc model load\n", __func__);
        return;
    }

    if (params.moe_hot_cache_path == nullptr || params.moe_hot_cache_path[0] == '\0') {
        throw std::runtime_error("--moe-hot-cache is required when --moe-hot-cache-max-mib is not 0");
    }

    LLAMA_LOG_WARN("%s: building hot-cache: max_mib = %lld, n_ctx = %u, n_seq_max = %u, n_ubatch = %u, swa_full = %d, kv_unified = %d, offload_kqv = %d, reserve_kv_cache = %d, path = %s\n",
            __func__,
            (long long) params.moe_hot_cache_max_mib,
            params.moe_hot_cache_auto_n_ctx,
            params.moe_hot_cache_auto_n_seq_max,
            params.moe_hot_cache_auto_n_ubatch,
            params.moe_hot_cache_auto_swa_full ? 1 : 0,
            params.moe_hot_cache_auto_kv_unified ? 1 : 0,
            params.moe_hot_cache_auto_offload_kqv ? 1 : 0,
            reserve_kv_cache ? 1 : 0,
            params.moe_hot_cache_path);

    std::ifstream file(params.moe_hot_cache_path);
    if (!file) {
        throw std::runtime_error(std::string("failed to open --moe-hot-cache file: ") + params.moe_hot_cache_path);
    }

    const std::string json_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const auto observations = llama_moe_hot_cache_parse_perf_json_observations(json_str);
    const auto config = weighting_config_from_params(&params);
    LLAMA_LOG_WARN("%s: MoE hot-cache weighting = %s, layer_curve = %.2f\n",
            __func__,
            llama_moe_hot_cache_weighting::mode_name(config.mode),
            config.layer_curve);
    const auto observed = score_observations_for_arch(model.arch, observations, &params);
    const auto sizes = llama_moe_hot_cache_collect_expert_sizes(model);
    ggml_backend_dev_t cache_dev = llama_moe_hot_cache_select_gpu_dev(&model);
    const size_t budget_bytes = params.moe_hot_cache_max_mib < 0
        ? llama_moe_hot_cache_auto_budget_bytes(model, params, cache_dev, reserve_kv_cache)
        : size_t(params.moe_hot_cache_max_mib)*LLAMA_MOE_HOT_CACHE_MIB;

    if (budget_bytes == 0) {
        LLAMA_LOG_WARN("%s: hot-cache budget is 0 MiB; disabling hot-cache\n", __func__);
        return;
    }

    const auto plan = llama_moe_hot_cache_select(observed, sizes, budget_bytes);

    const uint32_t n_expert_per_layer = model.hparams.n_expert;
    if (n_expert_per_layer > 0) {
        const double cpu_moe_layer_equiv = (double) plan.selected.size() / (double) n_expert_per_layer;
        LLAMA_LOG_WARN("%s: selected %zu/%zu observed experts for hot-cache (n-cpu-moe equivalent = %.1f layers @ %u experts/layer, %zu/%zu MiB)\n",
                __func__, plan.selected.size(), plan.observed.size(),
                cpu_moe_layer_equiv, n_expert_per_layer,
                plan.used_bytes/LLAMA_MOE_HOT_CACHE_MIB, plan.budget_bytes/LLAMA_MOE_HOT_CACHE_MIB);
    } else {
        LLAMA_LOG_WARN("%s: selected %zu/%zu observed experts for hot-cache (%zu/%zu MiB)\n",
                __func__, plan.selected.size(), plan.observed.size(),
                plan.used_bytes/LLAMA_MOE_HOT_CACHE_MIB, plan.budget_bytes/LLAMA_MOE_HOT_CACHE_MIB);
    }

    if (plan.selected.empty()) {
        LLAMA_LOG_WARN("%s: no experts selected; disabling hot-cache\n", __func__);
        return;
    }

    auto cache = std::make_unique<llama_moe_hot_cache>();
    cache->layers.resize(model.hparams.n_layer);

    std::vector<std::vector<uint32_t>> selected_by_layer(model.hparams.n_layer);
    for (const auto & selected : plan.selected) {
        if (selected.layer < selected_by_layer.size()) {
            selected_by_layer[selected.layer].push_back(selected.expert);
        }
    }

    size_t active_layers = 0;
    size_t total_hot_per_layer = 0;
    size_t min_hot_per_layer = std::numeric_limits<size_t>::max();
    size_t max_hot_per_layer = 0;

    for (const auto & experts : selected_by_layer) {
        if (experts.empty()) {
            continue;
        }

        active_layers++;
        total_hot_per_layer += experts.size();
        min_hot_per_layer = std::min(min_hot_per_layer, experts.size());
        max_hot_per_layer = std::max(max_hot_per_layer, experts.size());
    }

    LLAMA_LOG_INFO("%s: hot-cache active layers = %zu/%zu, hot experts per active layer min/avg/max = %zu/%.1f/%zu\n",
            __func__,
            active_layers,
            selected_by_layer.size(),
            active_layers > 0 ? min_hot_per_layer : 0,
            active_layers > 0 ? (double) total_hot_per_layer / (double) active_layers : 0.0,
            max_hot_per_layer);

    size_t n_tensors = 0;
    for (uint32_t il = 0; il < selected_by_layer.size(); ++il) {
        if (selected_by_layer[il].empty()) {
            continue;
        }

        const auto & layer = model.layers[il];
        const int64_t n_cache = int64_t(selected_by_layer[il].size()) + 1; // final entry is a zero dummy expert

        n_tensors += 4; // map + hot mask + cold mask + down
        n_tensors += layer.ffn_gate_up_exps != nullptr ? 1 : 2;
        n_tensors += layer.ffn_gate_exps_s != nullptr ? 1 : 0;
        n_tensors += layer.ffn_up_exps_s   != nullptr ? 1 : 0;
        n_tensors += layer.ffn_down_exps_s != nullptr ? 1 : 0;

        cache->layers[il].n_hot = selected_by_layer[il].size();
        cache->layers[il].n_expert = layer.ffn_down_exps ? layer.ffn_down_exps->ne[2] : 0;
        cache->layers[il].expert_weights_scale = model.hparams.expert_weights_scale;

        GGML_UNUSED(n_cache);
    }

    // Reserve extra space per tensor for internal ggml bookkeeping
    // (hash map entries, linked-list pointers, etc. beyond ggml_tensor_overhead())
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

            copy_expert_slice(src.ffn_gate_up_exps, dst.ffn_gate_up_exps, expert, cache_id);
            copy_expert_slice(src.ffn_gate_exps,    dst.ffn_gate_exps,    expert, cache_id);
            copy_expert_slice(src.ffn_up_exps,      dst.ffn_up_exps,      expert, cache_id);
            copy_expert_slice(src.ffn_down_exps,    dst.ffn_down_exps,    expert, cache_id);
            copy_scale_slice(src.ffn_gate_exps_s,   dst.ffn_gate_exps_s,  expert, cache_id);
            copy_scale_slice(src.ffn_up_exps_s,     dst.ffn_up_exps_s,    expert, cache_id);
            copy_scale_slice(src.ffn_down_exps_s,   dst.ffn_down_exps_s,  expert, cache_id);
        }

        ggml_backend_tensor_set(dst.hot_id_map, hot_id_map.data(), 0, hot_id_map.size()*sizeof(hot_id_map[0]));
        ggml_backend_tensor_set(dst.hot_mask,   hot_mask.data(),   0, hot_mask.size()*sizeof(hot_mask[0]));
        ggml_backend_tensor_set(dst.cold_mask,  cold_mask.data(),  0, cold_mask.size()*sizeof(cold_mask[0]));
    }

    LLAMA_LOG_WARN("%s: %12s hot-cache buffer size = %8.2f MiB\n",
            __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get())/1024.0/1024.0);

    cache->bufs.emplace_back(std::move(buf));
    cache->ctxs.emplace_back(std::move(ctx));
    model.moe_hot_cache = std::move(cache);
}

void llama_moe_hot_cache_init_after_model_load(llama_model & model, const llama_model_params & params) {
    if (params.moe_hot_cache_max_mib <= 0) {
        return;
    }

    llama_moe_hot_cache_init(model, params, true);
}

void llama_moe_hot_cache_init_after_context_memory(const llama_model & model) {
    const auto & params = model.get_params();
    if (model.hparams.vocab_only || params.moe_hot_cache_max_mib >= 0 || model.moe_hot_cache != nullptr) {
        return;
    }

    llama_moe_hot_cache_init(const_cast<llama_model &>(model), params, false);
}

llama_moe_hot_cache_update_stats llama_moe_hot_cache_update_from_perf_json(
        llama_model & model,
        const std::string & json_str,
        double update_rate) {
    llama_moe_hot_cache_update_stats stats;
    stats.update_rate = std::clamp(update_rate, 0.0, 1.0);

    if (!model.moe_hot_cache || !model.moe_hot_cache->active()) {
        return stats;
    }

    std::vector<llama_moe_hot_cache_perf_json_layer_slots> layer_slots;
    try {
        if (!llama_moe_hot_cache_perf_json_parser::parse_enabled_layer_slots(json_str, layer_slots)) {
            return stats;
        }
    } catch (const std::exception & e) {
        LLAMA_LOG_WARN("%s: failed to parse MoE layer perf JSON: %s\n", __func__, e.what());
        return stats;
    }

    stats.active = true;

    struct layer_counts {
        std::unordered_map<uint32_t, uint64_t> experts;
    };

    std::vector<layer_counts> counts(model.hparams.n_layer);

    std::vector<llama_moe_hot_cache_entry> scored_observed;
    try {
        const auto observations = llama_moe_hot_cache_parse_perf_json_observations(json_str);
        scored_observed = score_observations_for_arch(model.arch, observations, &model.get_params());
    } catch (const std::exception & e) {
        LLAMA_LOG_WARN("%s: failed to score MoE layer perf JSON: %s\n", __func__, e.what());
        return stats;
    }

    for (const auto & entry : scored_observed) {
        if (entry.layer < counts.size() && entry.hit_count > 0) {
            add_saturating(counts[entry.layer].experts[entry.expert], entry.hit_count);
        }
    }

    for (const auto & layer : layer_slots) {
        if (layer.layer >= counts.size()) {
            continue;
        }

        stats.hot_slots += layer.hot_slots;
        stats.cold_slots += layer.cold_slots;
    }

    const uint64_t total_slots = stats.hot_slots + stats.cold_slots;
    stats.hit_rate = total_slots > 0 ? (double) stats.hot_slots / (double) total_slots : 0.0;

    struct replacement_candidate {
        uint32_t layer = 0;
        uint32_t evict_expert = 0;
        uint32_t add_expert = 0;
        uint32_t cache_id = 0;
        uint64_t add_score = 0;
        uint64_t evict_score = 0;
    };

    std::vector<replacement_candidate> candidates;

    for (uint32_t il = 0; il < model.moe_hot_cache->layers.size(); ++il) {
        auto & cache_layer = model.moe_hot_cache->layers[il];
        if (!cache_layer.active() || il >= counts.size() || counts[il].experts.empty()) {
            continue;
        }

        std::vector<uint32_t> current = current_hot_experts(cache_layer);
        if (current.empty()) {
            continue;
        }

        stats.hot_experts += current.size();

        std::unordered_set<uint32_t> current_set(current.begin(), current.end());
        std::vector<std::pair<uint32_t, uint64_t>> ranked;
        ranked.reserve(counts[il].experts.size());
        for (const auto & [expert, hits] : counts[il].experts) {
            if (expert < cache_layer.n_expert && hits > 0) {
                ranked.push_back({ expert, hits });
            }
        }

        std::sort(ranked.begin(), ranked.end(), [](const auto & a, const auto & b) {
            if (a.second != b.second) {
                return a.second > b.second;
            }
            return a.first < b.first;
        });

        if (ranked.size() > current.size()) {
            ranked.resize(current.size());
        }

        std::unordered_set<uint32_t> desired_set;
        desired_set.reserve(ranked.size());
        for (const auto & [expert, hits] : ranked) {
            GGML_UNUSED(hits);
            desired_set.insert(expert);
        }

        std::vector<std::pair<uint32_t, uint64_t>> to_add;
        for (const auto & [expert, hits] : ranked) {
            if (current_set.find(expert) == current_set.end()) {
                to_add.push_back({ expert, hits });
            }
        }

        std::vector<std::pair<uint32_t, uint64_t>> to_evict;
        for (uint32_t expert : current) {
            if (desired_set.find(expert) == desired_set.end()) {
                const auto it = counts[il].experts.find(expert);
                to_evict.push_back({ expert, it == counts[il].experts.end() ? 0 : it->second });
            }
        }

        std::sort(to_evict.begin(), to_evict.end(), [](const auto & a, const auto & b) {
            if (a.second != b.second) {
                return a.second < b.second;
            }
            return a.first < b.first;
        });

        const size_t n_layer_candidates = std::min(to_add.size(), to_evict.size());
        for (size_t i = 0; i < n_layer_candidates; ++i) {
            const uint32_t evict = to_evict[i].first;
            const int32_t cache_id = cache_layer.hot_id_map_host[evict];
            if (cache_id < 0 || uint32_t(cache_id) >= cache_layer.n_hot) {
                continue;
            }

            candidates.push_back({
                    il,
                    evict,
                    to_add[i].first,
                    uint32_t(cache_id),
                    to_add[i].second,
                    to_evict[i].second,
                });
        }
    }

    stats.candidates = candidates.size();
    stats.max_exchange = stats.update_rate > 0.0 && stats.hot_experts > 0
        ? std::min(candidates.size(), (size_t) std::ceil(stats.update_rate * (double) stats.hot_experts))
        : 0;

    if (stats.max_exchange == 0) {
        return stats;
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto & a, const auto & b) {
        const uint64_t gain_a = a.add_score > a.evict_score ? a.add_score - a.evict_score : 0;
        const uint64_t gain_b = b.add_score > b.evict_score ? b.add_score - b.evict_score : 0;
        if (gain_a != gain_b) {
            return gain_a > gain_b;
        }
        if (a.layer != b.layer) {
            return a.layer < b.layer;
        }
        return a.add_expert < b.add_expert;
    });

    std::unordered_set<uint32_t> changed_layers;
    for (size_t i = 0; i < stats.max_exchange; ++i) {
        const auto & candidate = candidates[i];
        const auto & src = model.layers[candidate.layer];
        auto & dst = model.moe_hot_cache->layers[candidate.layer];

        if (candidate.add_expert >= dst.n_expert || candidate.evict_expert >= dst.n_expert) {
            continue;
        }

        const int32_t current_cache_id = dst.hot_id_map_host[candidate.evict_expert];
        if (current_cache_id < 0 || uint32_t(current_cache_id) != candidate.cache_id) {
            continue;
        }
        if (dst.hot_id_map_host[candidate.add_expert] >= 0) {
            continue;
        }

        copy_expert_slice(src.ffn_gate_up_exps, dst.ffn_gate_up_exps, candidate.add_expert, candidate.cache_id);
        copy_expert_slice(src.ffn_gate_exps,    dst.ffn_gate_exps,    candidate.add_expert, candidate.cache_id);
        copy_expert_slice(src.ffn_up_exps,      dst.ffn_up_exps,      candidate.add_expert, candidate.cache_id);
        copy_expert_slice(src.ffn_down_exps,    dst.ffn_down_exps,    candidate.add_expert, candidate.cache_id);
        copy_scale_slice(src.ffn_gate_exps_s,   dst.ffn_gate_exps_s,  candidate.add_expert, candidate.cache_id);
        copy_scale_slice(src.ffn_up_exps_s,     dst.ffn_up_exps_s,    candidate.add_expert, candidate.cache_id);
        copy_scale_slice(src.ffn_down_exps_s,   dst.ffn_down_exps_s,  candidate.add_expert, candidate.cache_id);

        dst.hot_id_map_host[candidate.evict_expert] = -1;
        dst.hot_id_map_host[candidate.add_expert] = int32_t(candidate.cache_id);

        set_tensor_i32_1d(dst.hot_id_map, candidate.evict_expert, int32_t(dst.n_hot));
        set_tensor_i32_1d(dst.hot_id_map, candidate.add_expert, int32_t(candidate.cache_id));
        set_tensor_f32_1d(dst.hot_mask, candidate.evict_expert, 0.0f);
        set_tensor_f32_1d(dst.hot_mask, candidate.add_expert, 1.0f);
        set_tensor_f32_1d(dst.cold_mask, candidate.evict_expert, 1.0f);
        set_tensor_f32_1d(dst.cold_mask, candidate.add_expert, 0.0f);

        stats.exchanged++;
        changed_layers.insert(candidate.layer);
    }

    stats.layers_changed = changed_layers.size();
    return stats;
}

bool llama_moe_hot_cache_layer_active(const llama_model & model, int il) {
    return model.moe_hot_cache &&
           il >= 0 &&
           il < int(model.moe_hot_cache->layers.size()) &&
           model.moe_hot_cache->layers[il].active();
}

void llama_moe_hot_cache_build_worklist(
        ggml_tensor * dst,
        const ggml_tensor * selected_experts,
        const ggml_tensor * weights,
        const llama_moe_hot_cache_layer & layer,
        int ith,
        int nth) {
    GGML_UNUSED(nth);

    if (ith != 0) {
        return;
    }

    GGML_ASSERT(dst != nullptr);
    GGML_ASSERT(selected_experts != nullptr);
    GGML_ASSERT(weights != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(selected_experts->type == GGML_TYPE_I32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[1] == LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COUNT);
    GGML_ASSERT(weights->ne[0] == 1);
    GGML_ASSERT(selected_experts->ne[0] == weights->ne[1]);
    GGML_ASSERT(selected_experts->ne[1] == weights->ne[2]);
    GGML_ASSERT(int64_t(layer.hot_id_map_host.size()) == layer.n_expert);

    const int32_t capacity = dst->ne[0];
    const int32_t n_expert_used = selected_experts->ne[0];
    const int32_t n_tokens = selected_experts->ne[1];
    const int32_t total_slots = n_expert_used * n_tokens;
    const int32_t dummy_src_slot = total_slots;
    const float hot_padding_id =
        llama_moe_hot_cache_hot_dummy_padding() && layer.n_hot > 0 ? float(layer.n_hot) : -1.0f;

    GGML_ASSERT(capacity == total_slots);

    auto set_field = [&](int32_t field, int32_t slot, float value) {
        char * row = (char *) dst->data + field*dst->nb[1];
        *(float *)(row + slot*dst->nb[0]) = value;
    };

    for (int32_t slot = 0; slot < capacity; ++slot) {
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID,        slot, hot_padding_id);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT,  slot, float(dummy_src_slot));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID,  slot, 0.0f);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT,    slot, 0.0f);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID,       slot, -1.0f);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT, slot, float(dummy_src_slot));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, slot, 0.0f);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT,   slot, 0.0f);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, slot, -1.0f);
    }

    int32_t hot_slot = 0;
    int32_t cold_slot = 0;

    for (int32_t token = 0; token < n_tokens; ++token) {
        for (int32_t iex = 0; iex < n_expert_used; ++iex) {
            const int32_t expert = *(const int32_t *) ((const char *) selected_experts->data + iex*selected_experts->nb[0] + token*selected_experts->nb[1]);
            GGML_ASSERT(expert >= 0);
            GGML_ASSERT(expert < int32_t(layer.hot_id_map_host.size()));

            const float weight = *(const float *) ((const char *) weights->data + iex*weights->nb[1] + token*weights->nb[2]);
            const int32_t src_slot = token*n_expert_used + iex;
            const int32_t hot_id = layer.hot_id_map_host[expert];

            if (hot_id >= 0) {
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID,        hot_slot, float(hot_id));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT,  hot_slot, float(src_slot));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID,  hot_slot, float(token));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT,    hot_slot, weight);
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, hot_slot, float(expert));
                ++hot_slot;
            } else {
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID,       cold_slot, float(expert));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT, cold_slot, float(src_slot));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, cold_slot, float(token));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT,   cold_slot, weight);
                ++cold_slot;
            }
        }
    }

    GGML_ASSERT(hot_slot  <= capacity);
    GGML_ASSERT(cold_slot <= capacity);

    if (capacity > 0) {
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_COUNT,  0, float(hot_slot));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_COUNT, 0, float(cold_slot));
    }
}

void llama_moe_hot_cache_build_worklist_from_logits(
        ggml_tensor * dst,
        const ggml_tensor * logits,
        const llama_moe_hot_cache_layer & layer,
        int ith,
        int nth) {
    GGML_UNUSED(nth);

    if (ith != 0) {
        return;
    }

    GGML_ASSERT(dst != nullptr);
    GGML_ASSERT(logits != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[1] == LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COUNT);
    GGML_ASSERT(int64_t(layer.hot_id_map_host.size()) == layer.n_expert);
    GGML_ASSERT(logits->ne[0] == layer.n_expert);

    const int32_t capacity = dst->ne[0];
    const int32_t n_expert = logits->ne[0];
    const int32_t n_tokens = logits->ne[1];
    GGML_ASSERT(n_tokens > 0);
    GGML_ASSERT(capacity % n_tokens == 0);

    const int32_t n_expert_used = capacity / n_tokens;
    const int32_t total_slots = n_expert_used * n_tokens;
    const int32_t dummy_src_slot = total_slots;
    const float hot_padding_id =
        llama_moe_hot_cache_hot_dummy_padding() && layer.n_hot > 0 ? float(layer.n_hot) : -1.0f;
    GGML_ASSERT(capacity == total_slots);
    GGML_ASSERT(n_expert_used > 0);
    GGML_ASSERT(n_expert_used <= n_expert);
    GGML_ASSERT(n_expert_used <= LLAMA_MAX_EXPERTS);

    auto set_field = [&](int32_t field, int32_t slot, float value) {
        char * row = (char *) dst->data + field*dst->nb[1];
        *(float *)(row + slot*dst->nb[0]) = value;
    };

    for (int32_t slot = 0; slot < capacity; ++slot) {
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID,        slot, hot_padding_id);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT,  slot, float(dummy_src_slot));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID,  slot, 0.0f);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT,    slot, 0.0f);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID,       slot, -1.0f);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT, slot, float(dummy_src_slot));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, slot, 0.0f);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT,   slot, 0.0f);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, slot, -1.0f);
    }

    int32_t hot_slot = 0;
    int32_t cold_slot = 0;
    int32_t top_experts[LLAMA_MAX_EXPERTS];
    float top_logits[LLAMA_MAX_EXPERTS];

    for (int32_t token = 0; token < n_tokens; ++token) {
        for (int32_t i = 0; i < n_expert_used; ++i) {
            top_experts[i] = -1;
            top_logits[i] = -std::numeric_limits<float>::infinity();
        }

        for (int32_t expert = 0; expert < n_expert; ++expert) {
            const float logit = *(const float *) ((const char *) logits->data + expert*logits->nb[0] + token*logits->nb[1]);

            for (int32_t pos = 0; pos < n_expert_used; ++pos) {
                if (logit <= top_logits[pos]) {
                    continue;
                }

                for (int32_t move = n_expert_used - 1; move > pos; --move) {
                    top_logits[move] = top_logits[move - 1];
                    top_experts[move] = top_experts[move - 1];
                }
                top_logits[pos] = logit;
                top_experts[pos] = expert;
                break;
            }
        }

        const float max_logit = top_logits[0];
        float weight_sum = 0.0f;
        for (int32_t iex = 0; iex < n_expert_used; ++iex) {
            top_logits[iex] = std::exp(top_logits[iex] - max_logit);
            weight_sum += top_logits[iex];
        }

        const float weight_scale =
            layer.expert_weights_scale != 0.0f && layer.expert_weights_scale != 1.0f ? layer.expert_weights_scale : 1.0f;

        for (int32_t iex = 0; iex < n_expert_used; ++iex) {
            const int32_t expert = top_experts[iex];
            GGML_ASSERT(expert >= 0);
            GGML_ASSERT(expert < int32_t(layer.hot_id_map_host.size()));

            const float weight = top_logits[iex] / weight_sum * weight_scale;
            const int32_t src_slot = token*n_expert_used + iex;
            const int32_t hot_id = layer.hot_id_map_host[expert];

            if (hot_id >= 0) {
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID,        hot_slot, float(hot_id));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT,  hot_slot, float(src_slot));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID,  hot_slot, float(token));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT,    hot_slot, weight);
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, hot_slot, float(expert));
                ++hot_slot;
            } else {
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID,       cold_slot, float(expert));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT, cold_slot, float(src_slot));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, cold_slot, float(token));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT,   cold_slot, weight);
                ++cold_slot;
            }
        }
    }

    GGML_ASSERT(hot_slot  <= capacity);
    GGML_ASSERT(cold_slot <= capacity);

    if (capacity > 0) {
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_COUNT,  0, float(hot_slot));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_COUNT, 0, float(cold_slot));
    }
}
