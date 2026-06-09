#include "llama-moe-hot-cache.h"
#include "llama-moe-hot-cache-adapter.h"
#include "llama-moe-hot-cache-budget.h"
#include "llama-moe-hot-cache-builder.h"
#include "llama-moe-hot-cache-common.h"
#include "llama-moe-hot-cache-parser.h"
#include "llama-moe-hot-cache-planner.h"
#include "llama-moe-hot-cache-updater.h"

#include "llama-model.h"
#include "llama-impl.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace {

static void add_saturating(uint64_t & dst, uint64_t value) {
    if (dst > std::numeric_limits<uint64_t>::max() - value) {
        dst = std::numeric_limits<uint64_t>::max();
    } else {
        dst += value;
    }
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

static bool hot_cache_lane_enabled(int64_t max_mib) {
    return max_mib != 0;
}

static bool hot_cache_any_lane_enabled(const llama_model_params & params) {
    return hot_cache_lane_enabled(params.moe_hot_cache_max_mib) ||
           hot_cache_lane_enabled(params.moe_hot_cache_second_max_mib) ||
           hot_cache_lane_enabled(params.moe_hot_cache_third_max_mib);
}

static bool hot_cache_any_static_lane_enabled(const llama_model_params & params) {
    return params.moe_hot_cache_max_mib > 0 ||
           params.moe_hot_cache_second_max_mib > 0 ||
           params.moe_hot_cache_third_max_mib > 0;
}

static bool hot_cache_any_auto_lane_enabled(const llama_model_params & params) {
    return params.moe_hot_cache_max_mib < 0 ||
           params.moe_hot_cache_second_max_mib < 0 ||
           params.moe_hot_cache_third_max_mib < 0;
}

struct hot_cache_lane_request {
    const char * name = nullptr;
    int64_t max_mib = 0;
    uint64_t reserve_mib = 0;
};

static std::vector<hot_cache_lane_request> hot_cache_lane_requests(const llama_model_params & params) {
    std::vector<hot_cache_lane_request> lanes;
    if (hot_cache_lane_enabled(params.moe_hot_cache_max_mib)) {
        lanes.push_back({ params.moe_hot_cache_device, params.moe_hot_cache_max_mib, params.moe_hot_cache_auto_reserve_mib });
    }
    if (hot_cache_lane_enabled(params.moe_hot_cache_second_max_mib)) {
        lanes.push_back({ params.moe_hot_cache_second_device, params.moe_hot_cache_second_max_mib, params.moe_hot_cache_second_auto_reserve_mib });
    }
    if (hot_cache_lane_enabled(params.moe_hot_cache_third_max_mib)) {
        lanes.push_back({ params.moe_hot_cache_third_device, params.moe_hot_cache_third_max_mib, params.moe_hot_cache_third_auto_reserve_mib });
    }
    return lanes;
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
    if (!hot_cache_any_lane_enabled(params)) {
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

    LLAMA_LOG_WARN("%s: building hot-cache: max_mib = %lld, second_max_mib = %lld, third_max_mib = %lld, n_ctx = %u, n_seq_max = %u, n_ubatch = %u, swa_full = %d, kv_unified = %d, offload_kqv = %d, reserve_kv_cache = %d, path = %s\n",
            __func__,
            (long long) params.moe_hot_cache_max_mib,
            (long long) params.moe_hot_cache_second_max_mib,
            (long long) params.moe_hot_cache_third_max_mib,
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

    const auto lane_requests = hot_cache_lane_requests(params);
    if (lane_requests.empty() || lane_requests.size() > LLAMA_MOE_HOT_CACHE_MAX_EXPERT_LANES) {
        LLAMA_LOG_WARN("%s: no hot-cache lanes configured; disabling hot-cache\n", __func__);
        return;
    }

    std::vector<ggml_backend_dev_t> cache_devs;
    std::vector<size_t> lane_budgets;
    cache_devs.reserve(lane_requests.size());
    lane_budgets.reserve(lane_requests.size());

    std::unordered_set<ggml_backend_dev_t> seen_devs;
    for (size_t lane = 0; lane < lane_requests.size(); ++lane) {
        const auto & request = lane_requests[lane];
        ggml_backend_dev_t dev = llama_moe_hot_cache_resolve_gpu_dev(&model, request.name);
        if (!seen_devs.insert(dev).second) {
            throw std::runtime_error(std::string("duplicate MoE hot-cache expert device: ") + ggml_backend_dev_name(dev));
        }

        const size_t budget_bytes = request.max_mib < 0
            ? llama_moe_hot_cache_auto_budget_bytes(model, params, dev, reserve_kv_cache, request.reserve_mib)
            : size_t(request.max_mib)*LLAMA_MOE_HOT_CACHE_MIB;

        LLAMA_LOG_WARN("%s: expert lane %zu device = %s, max_mib = %lld, reserve_mib = %zu, budget = %zu MiB\n",
                __func__,
                lane,
                ggml_backend_dev_name(dev),
                (long long) request.max_mib,
                (size_t) request.reserve_mib,
                budget_bytes/LLAMA_MOE_HOT_CACHE_MIB);

        cache_devs.push_back(dev);
        lane_budgets.push_back(budget_bytes);
    }

    const auto strategy = llama_moe_hot_cache_parse_device_strategy(params.moe_hot_cache_device_strategy);
    const auto plan = llama_moe_hot_cache_select_multi(observed, sizes, lane_budgets, strategy);

    const uint32_t n_expert_per_layer = model.hparams.n_expert;
    if (n_expert_per_layer > 0) {
        const double cpu_moe_layer_equiv = (double) plan.selected_count() / (double) n_expert_per_layer;
        LLAMA_LOG_WARN("%s: selected %zu/%zu observed experts for hot-cache across %zu lanes (n-cpu-moe equivalent = %.1f layers @ %u experts/layer, %zu/%zu MiB)\n",
                __func__, plan.selected_count(), plan.observed.size(), plan.lanes.size(),
                cpu_moe_layer_equiv, n_expert_per_layer,
                plan.used_bytes()/LLAMA_MOE_HOT_CACHE_MIB, plan.budget_bytes()/LLAMA_MOE_HOT_CACHE_MIB);
    } else {
        LLAMA_LOG_WARN("%s: selected %zu/%zu observed experts for hot-cache across %zu lanes (%zu/%zu MiB)\n",
                __func__, plan.selected_count(), plan.observed.size(), plan.lanes.size(),
                plan.used_bytes()/LLAMA_MOE_HOT_CACHE_MIB, plan.budget_bytes()/LLAMA_MOE_HOT_CACHE_MIB);
    }
    for (size_t lane = 0; lane < plan.lanes.size(); ++lane) {
        const auto & lane_plan = plan.lanes[lane];
        LLAMA_LOG_WARN("%s: expert lane %zu plan on %s: selected = %zu, used = %zu/%zu MiB\n",
                __func__,
                lane,
                ggml_backend_dev_name(cache_devs.at(lane)),
                lane_plan.selected.size(),
                lane_plan.used_bytes/LLAMA_MOE_HOT_CACHE_MIB,
                lane_plan.budget_bytes/LLAMA_MOE_HOT_CACHE_MIB);
    }

    if (plan.selected_count() == 0) {
        LLAMA_LOG_WARN("%s: no experts selected; disabling hot-cache\n", __func__);
        return;
    }

    model.moe_hot_cache = llama_moe_hot_cache_build_multi(model, plan, cache_devs);
}

void llama_moe_hot_cache_init_after_model_load(llama_model & model, const llama_model_params & params) {
    if (!hot_cache_any_static_lane_enabled(params)) {
        return;
    }

    llama_moe_hot_cache_init(model, params, true);
}

void llama_moe_hot_cache_init_after_context_memory(const llama_model & model) {
    const auto & params = model.get_params();
    if (model.hparams.vocab_only || !hot_cache_any_auto_lane_enabled(params) || model.moe_hot_cache != nullptr) {
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

    std::vector<llama_moe_hot_cache_entry> scored_observed;
    try {
        const auto observations = llama_moe_hot_cache_parse_perf_json_observations(json_str);
        scored_observed = score_observations_for_arch(model.arch, observations, &model.get_params());
    } catch (const std::exception & e) {
        LLAMA_LOG_WARN("%s: failed to score MoE layer perf JSON: %s\n", __func__, e.what());
        return stats;
    }

    return llama_moe_hot_cache_update_from_scored_observations(
            model,
            layer_slots,
            scored_observed,
            stats.update_rate);
}

llama_moe_hot_cache_update_stats llama_moe_hot_cache_apply_json(
        llama_model & model,
        const std::string & json_str) {
    llama_moe_hot_cache_update_stats stats;
    stats.update_rate = 1.0;

    if (!model.moe_hot_cache || !model.moe_hot_cache->active()) {
        return stats;
    }

    std::vector<llama_moe_hot_cache_entry> scored_observed;
    try {
        const auto observations = llama_moe_hot_cache_parse_perf_json_observations(json_str);
        scored_observed = score_observations_for_arch(model.arch, observations, &model.get_params());
    } catch (const std::exception & e) {
        LLAMA_LOG_WARN("%s: failed to parse MoE hot-cache JSON: %s\n", __func__, e.what());
        return stats;
    }

    stats.active = true;

    return llama_moe_hot_cache_update_from_scored_observations(
            model,
            {},
            scored_observed,
            stats.update_rate);
}

bool llama_moe_hot_cache_layer_active(const llama_model & model, int il) {
    return llama_moe_hot_cache_layer_active_for_graph(model, il, llama_moe_hot_cache_graph_kind::none);
}

bool llama_moe_hot_cache_layer_active_for_graph(
        const llama_model & model,
        int il,
        llama_moe_hot_cache_graph_kind graph_kind) {
    return llama_moe_hot_cache_adapter_supports_graph_kind(model.arch, graph_kind) &&
           model.moe_hot_cache &&
           il >= 0 &&
           il < int(model.moe_hot_cache->layers.size()) &&
           model.moe_hot_cache->layers[il].active();
}
