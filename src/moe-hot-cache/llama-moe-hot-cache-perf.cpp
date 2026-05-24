#include "llama-moe-hot-cache-perf.h"
#include "llama-moe-hot-cache-perf-json.h"
#include "llama-moe-hot-cache-perf-nodes.h"

#include "ggml.h"
#include "ggml-backend-moe-hot-cache.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

struct llama_moe_layer_perf_layer {
    uint64_t calls = 0;
    uint64_t expert_hits_total = 0;
    uint64_t hot_slots_total = 0;
    uint64_t cold_slots_total = 0;
    uint64_t hot_worklist_calls = 0;
    uint64_t cold_worklist_calls = 0;
    uint64_t hot_zero_calls = 0;
    uint64_t cold_zero_calls = 0;

    uint64_t total_moe_time_us = 0;
    uint64_t expert_matmul_time_us = 0;
    uint64_t hot_branch_time_us = 0;
    uint64_t cold_branch_time_us = 0;
    uint64_t hot_expert_matmul_time_us = 0;
    uint64_t cold_expert_matmul_time_us = 0;
    uint64_t worklist_time_us = 0;
    uint64_t routing_time_us = 0;
    uint64_t merge_time_us = 0;
    uint64_t hot_gather_scatter_time_us = 0;
    uint64_t cold_gather_scatter_time_us = 0;

    uint64_t parallel_region_wall_time_us = 0;
    uint64_t parallel_hot_lane_wall_time_us = 0;
    uint64_t parallel_cold_lane_wall_time_us = 0;
    uint64_t parallel_join_wait_time_us = 0;
    uint64_t parallel_overlap_estimate_us = 0;
    uint64_t parallel_regions = 0;
    uint64_t parallel_hot_launches = 0;
    uint64_t parallel_cold_launches = 0;
    uint64_t parallel_hot_skips_zero = 0;
    uint64_t parallel_cold_skips_zero = 0;
    uint64_t parallel_fallbacks = 0;
    uint64_t parallel_fallback_incomplete = 0;
    uint64_t parallel_fallback_count_not_prefix = 0;
    uint64_t parallel_fallback_bad_split_order = 0;
    uint64_t parallel_fallback_same_backend = 0;
    uint64_t parallel_fallback_hot_spans_backends = 0;
    uint64_t parallel_fallback_cold_spans_backends = 0;
    uint64_t parallel_fallback_hot_not_cuda = 0;
    uint64_t parallel_fallback_cold_not_cpu = 0;
    uint64_t parallel_fallback_count_readback = 0;
    uint64_t parallel_fallback_threshold = 0;
    uint64_t parallel_fallback_zero_output = 0;
    uint64_t parallel_fallback_other = 0;
    uint64_t parallel_split_debug_samples = 0;
    int32_t parallel_split_debug_hot_begin = -1;
    int32_t parallel_split_debug_hot_end = -1;
    int32_t parallel_split_debug_cold_begin = -1;
    int32_t parallel_split_debug_cold_end = -1;
    int32_t parallel_split_debug_join = -1;
    int32_t parallel_split_debug_hot_count = -1;
    int32_t parallel_split_debug_cold_count = -1;
    int32_t parallel_split_debug_hot_backend = -1;
    int32_t parallel_split_debug_cold_backend = -1;
    int32_t parallel_split_debug_join_backend = -1;

    uint64_t gate_time_us = 0;
    uint64_t up_time_us = 0;
    uint64_t down_time_us = 0;

    std::vector<uint64_t> experts;
    std::vector<uint64_t> hot_experts;
    std::vector<uint64_t> cold_experts;
};

struct llama_moe_layer_perf_local {
    std::mutex mutex;

    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;

    uint64_t updates = 0;
    uint64_t overflow_resets = 0;

    bool active = false;
    int64_t last_callback_us = 0;

    std::vector<llama_moe_layer_perf_layer> layers;

    void ensure_shape_locked(uint32_t n_layer, uint32_t experts, uint32_t used) {
        n_expert = experts;
        n_expert_used = used;

        if (layers.size() < n_layer) {
            layers.resize(n_layer);
        }

        for (auto & layer : layers) {
            if (layer.experts.size() < experts) {
                layer.experts.resize(experts, 0);
            }
            if (layer.hot_experts.size() < experts) {
                layer.hot_experts.resize(experts, 0);
            }
            if (layer.cold_experts.size() < experts) {
                layer.cold_experts.resize(experts, 0);
            }
        }
    }

    void reset_locked(bool count_overflow = true) {
        for (auto & layer : layers) {
            layer.calls = 0;
            layer.expert_hits_total = 0;
            layer.hot_slots_total = 0;
            layer.cold_slots_total = 0;
            layer.hot_worklist_calls = 0;
            layer.cold_worklist_calls = 0;
            layer.hot_zero_calls = 0;
            layer.cold_zero_calls = 0;
            layer.total_moe_time_us = 0;
            layer.expert_matmul_time_us = 0;
            layer.hot_branch_time_us = 0;
            layer.cold_branch_time_us = 0;
            layer.hot_expert_matmul_time_us = 0;
            layer.cold_expert_matmul_time_us = 0;
            layer.worklist_time_us = 0;
            layer.routing_time_us = 0;
            layer.merge_time_us = 0;
            layer.hot_gather_scatter_time_us = 0;
            layer.cold_gather_scatter_time_us = 0;
            layer.parallel_region_wall_time_us = 0;
            layer.parallel_hot_lane_wall_time_us = 0;
            layer.parallel_cold_lane_wall_time_us = 0;
            layer.parallel_join_wait_time_us = 0;
            layer.parallel_overlap_estimate_us = 0;
            layer.parallel_regions = 0;
            layer.parallel_hot_launches = 0;
            layer.parallel_cold_launches = 0;
            layer.parallel_hot_skips_zero = 0;
            layer.parallel_cold_skips_zero = 0;
            layer.parallel_fallbacks = 0;
            layer.parallel_fallback_incomplete = 0;
            layer.parallel_fallback_count_not_prefix = 0;
            layer.parallel_fallback_bad_split_order = 0;
            layer.parallel_fallback_same_backend = 0;
            layer.parallel_fallback_hot_spans_backends = 0;
            layer.parallel_fallback_cold_spans_backends = 0;
            layer.parallel_fallback_hot_not_cuda = 0;
            layer.parallel_fallback_cold_not_cpu = 0;
            layer.parallel_fallback_count_readback = 0;
            layer.parallel_fallback_threshold = 0;
            layer.parallel_fallback_zero_output = 0;
            layer.parallel_fallback_other = 0;
            layer.parallel_split_debug_samples = 0;
            layer.parallel_split_debug_hot_begin = -1;
            layer.parallel_split_debug_hot_end = -1;
            layer.parallel_split_debug_cold_begin = -1;
            layer.parallel_split_debug_cold_end = -1;
            layer.parallel_split_debug_join = -1;
            layer.parallel_split_debug_hot_count = -1;
            layer.parallel_split_debug_cold_count = -1;
            layer.parallel_split_debug_hot_backend = -1;
            layer.parallel_split_debug_cold_backend = -1;
            layer.parallel_split_debug_join_backend = -1;
            layer.gate_time_us = 0;
            layer.up_time_us = 0;
            layer.down_time_us = 0;
            std::fill(layer.experts.begin(), layer.experts.end(), 0);
            std::fill(layer.hot_experts.begin(), layer.hot_experts.end(), 0);
            std::fill(layer.cold_experts.begin(), layer.cold_experts.end(), 0);
        }

        updates = 0;
        if (count_overflow) {
            overflow_resets++;
        }
        last_callback_us = 0;
    }

    void add_locked(uint64_t & dst, uint64_t add) {
        if (dst > std::numeric_limits<uint64_t>::max() - add) {
            reset_locked();
        }

        dst += add;
    }

    void add_expert_locked(uint32_t layer, uint32_t expert) {
        if (layer >= layers.size()) {
            return;
        }

        if (expert >= layers[layer].experts.size()) {
            return;
        }

        add_locked(layers[layer].experts[expert], 1);
        add_locked(layers[layer].expert_hits_total, 1);
    }

    void add_branch_expert_locked(uint32_t layer, uint32_t expert, bool hot) {
        if (layer >= layers.size()) {
            return;
        }

        auto & counts = hot ? layers[layer].hot_experts : layers[layer].cold_experts;
        if (expert >= counts.size()) {
            return;
        }

        add_locked(counts[expert], 1);
    }
};

static llama_moe_layer_perf_local g_llama_moe_layer_perf;
static std::atomic<int> g_llama_moe_layer_perf_mode{-1};

bool llama_moe_layer_perf_parse_mode(const char * value, llama_moe_layer_perf_mode & mode) {
    if (value == nullptr) {
        return false;
    }

    if (std::strcmp(value, "full") == 0 || std::strcmp(value, "1") == 0 || std::strcmp(value, "on") == 0 || std::strcmp(value, "true") == 0) {
        mode = LLAMA_MOE_LAYER_PERF_MODE_FULL;
        return true;
    }

    if (std::strcmp(value, "update") == 0) {
        mode = LLAMA_MOE_LAYER_PERF_MODE_UPDATE;
        return true;
    }

    if (std::strcmp(value, "off") == 0 || std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0) {
        mode = LLAMA_MOE_LAYER_PERF_MODE_OFF;
        return true;
    }

    return false;
}

const char * llama_moe_layer_perf_mode_name(llama_moe_layer_perf_mode mode) {
    switch (mode) {
        case LLAMA_MOE_LAYER_PERF_MODE_FULL:
            return "full";
        case LLAMA_MOE_LAYER_PERF_MODE_UPDATE:
            return "update";
        case LLAMA_MOE_LAYER_PERF_MODE_OFF:
        default:
            return "off";
    }
}

static llama_moe_layer_perf_mode llama_moe_layer_perf_env_mode() {
    const char * env = std::getenv("LLAMA_MOE_LAYER_PERF");
    if (env == nullptr || env[0] == '\0') {
        return LLAMA_MOE_LAYER_PERF_MODE_FULL;
    }

    llama_moe_layer_perf_mode mode = LLAMA_MOE_LAYER_PERF_MODE_FULL;
    return llama_moe_layer_perf_parse_mode(env, mode) ? mode : LLAMA_MOE_LAYER_PERF_MODE_FULL;
}

llama_moe_layer_perf_mode llama_moe_layer_perf_get_mode(const llama_context * ctx) {
    const int configured = g_llama_moe_layer_perf_mode.load(std::memory_order_relaxed);
    if (configured >= 0) {
        return (llama_moe_layer_perf_mode) configured;
    }

    if (ctx != nullptr && ctx->get_cparams().no_perf) {
        return LLAMA_MOE_LAYER_PERF_MODE_OFF;
    }

    return llama_moe_layer_perf_env_mode();
}

void llama_moe_layer_perf_set_initial_mode(bool no_perf) {
    llama_moe_layer_perf_set_mode(no_perf ? LLAMA_MOE_LAYER_PERF_MODE_OFF : llama_moe_layer_perf_env_mode());
}

void llama_moe_layer_perf_set_mode(llama_moe_layer_perf_mode mode) {
    g_llama_moe_layer_perf_mode.store((int) mode, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lock(g_llama_moe_layer_perf.mutex);
    g_llama_moe_layer_perf.reset_locked(false);
    g_llama_moe_layer_perf.active = false;
    g_llama_moe_layer_perf.updates = 0;
}

bool llama_moe_layer_perf_is_enabled(const llama_context * ctx) {
    return llama_moe_layer_perf_get_mode(ctx) != LLAMA_MOE_LAYER_PERF_MODE_OFF;
}

bool llama_moe_layer_perf_needs_expert_counts(bool no_perf) {
    const int configured = g_llama_moe_layer_perf_mode.load(std::memory_order_relaxed);
    const llama_moe_layer_perf_mode mode = configured >= 0 ?
        (llama_moe_layer_perf_mode) configured :
        (no_perf ? LLAMA_MOE_LAYER_PERF_MODE_OFF : llama_moe_layer_perf_env_mode());

    return mode != LLAMA_MOE_LAYER_PERF_MODE_OFF;
}

void llama_moe_layer_perf_begin(uint32_t n_layer, uint32_t n_expert, uint32_t n_expert_used) {
    std::lock_guard<std::mutex> lock(g_llama_moe_layer_perf.mutex);

    if (n_layer == 0 || n_expert == 0 || n_expert_used == 0) {
        g_llama_moe_layer_perf.active = false;
        return;
    }

    g_llama_moe_layer_perf.ensure_shape_locked(n_layer, n_expert, n_expert_used);
    g_llama_moe_layer_perf.active = true;
    g_llama_moe_layer_perf.last_callback_us = 0;

    if (g_llama_moe_layer_perf.updates == std::numeric_limits<uint64_t>::max()) {
        g_llama_moe_layer_perf.reset_locked();
    }

    g_llama_moe_layer_perf.updates++;
}

void llama_moe_layer_perf_end() {
    std::lock_guard<std::mutex> lock(g_llama_moe_layer_perf.mutex);

    g_llama_moe_layer_perf.active = false;
    g_llama_moe_layer_perf.last_callback_us = 0;
}

void llama_moe_layer_perf_reset() {
    std::lock_guard<std::mutex> lock(g_llama_moe_layer_perf.mutex);

    g_llama_moe_layer_perf.reset_locked(false);
    g_llama_moe_layer_perf.updates = 0;
}

bool llama_moe_layer_perf_has_data() {
    std::lock_guard<std::mutex> lock(g_llama_moe_layer_perf.mutex);

    return std::any_of(
            g_llama_moe_layer_perf.layers.begin(),
            g_llama_moe_layer_perf.layers.end(),
            [](const llama_moe_layer_perf_layer & layer) {
                return layer.calls != 0 ||
                       layer.expert_hits_total != 0 ||
                       layer.total_moe_time_us != 0 ||
                       layer.parallel_fallbacks != 0;
            });
}

static void llama_moe_layer_perf_count_topk_locked(uint32_t layer, ggml_tensor * t) {
    if (t == nullptr) {
        return;
    }

    const int64_t k        = t->ne[0];
    const int64_t n_tokens = t->ne[1];

    if (k <= 0 || n_tokens <= 0) {
        return;
    }

    std::vector<int32_t> ids(k * n_tokens);

    ggml_backend_tensor_get(
        t,
        ids.data(),
        0,
        ids.size() * sizeof(int32_t));

    for (int64_t i = 0; i < k * n_tokens; ++i) {
        const int32_t expert = ids[i];

        if (expert >= 0 && (uint32_t) expert < g_llama_moe_layer_perf.n_expert) {
            g_llama_moe_layer_perf.add_expert_locked(layer, (uint32_t) expert);
        }
    }

    if (layer < g_llama_moe_layer_perf.layers.size()) {
        g_llama_moe_layer_perf.add_locked(g_llama_moe_layer_perf.layers[layer].calls, 1);
    }
}

static void llama_moe_layer_perf_count_worklist_count_locked(uint32_t layer, ggml_tensor * t, bool hot) {
    if (t == nullptr || layer >= g_llama_moe_layer_perf.layers.size()) {
        return;
    }

    if (ggml_nelements(t) <= 0) {
        return;
    }

    uint64_t valid = 0;
    if (t->type == GGML_TYPE_F32) {
        float count = 0.0f;
        ggml_backend_tensor_get(t, &count, 0, sizeof(count));
        valid = count > 0.0f ? (uint64_t) (count + 0.5f) : 0;
    } else if (t->type == GGML_TYPE_I32) {
        int32_t count = 0;
        ggml_backend_tensor_get(t, &count, 0, sizeof(count));
        valid = count > 0 ? (uint64_t) count : 0;
    } else {
        return;
    }

    auto & dst = g_llama_moe_layer_perf.layers[layer];
    if (hot) {
        g_llama_moe_layer_perf.add_locked(dst.hot_worklist_calls, 1);
        if (dst.calls < dst.hot_worklist_calls) {
            g_llama_moe_layer_perf.add_locked(dst.calls, 1);
        }
        g_llama_moe_layer_perf.add_locked(dst.hot_slots_total, valid);
        if (valid == 0) {
            g_llama_moe_layer_perf.add_locked(dst.hot_zero_calls, 1);
        }
    } else {
        g_llama_moe_layer_perf.add_locked(dst.cold_worklist_calls, 1);
        g_llama_moe_layer_perf.add_locked(dst.cold_slots_total, valid);
        if (valid == 0) {
            g_llama_moe_layer_perf.add_locked(dst.cold_zero_calls, 1);
        }
    }
}

static void llama_moe_layer_perf_count_branch_experts_locked(uint32_t layer, ggml_tensor * t, bool hot) {
    if (t == nullptr || layer >= g_llama_moe_layer_perf.layers.size()) {
        return;
    }

    const int64_t n_ids = ggml_nelements(t);
    if (n_ids <= 0) {
        return;
    }

    if (t->type == GGML_TYPE_I32) {
        std::vector<int32_t> ids(n_ids);
        ggml_backend_tensor_get(
            t,
            ids.data(),
            0,
            ids.size() * sizeof(int32_t));

        for (int32_t id : ids) {
            if (id >= 0 && (uint32_t) id < g_llama_moe_layer_perf.n_expert) {
                g_llama_moe_layer_perf.add_branch_expert_locked(layer, (uint32_t) id, hot);
            }
        }
    } else if (t->type == GGML_TYPE_F32) {
        std::vector<float> ids(n_ids);
        ggml_backend_tensor_get(
            t,
            ids.data(),
            0,
            ids.size() * sizeof(float));

        for (float idf : ids) {
            const int32_t id = (int32_t) (idf + 0.5f);
            if (idf >= 0.0f && id >= 0 && (uint32_t) id < g_llama_moe_layer_perf.n_expert) {
                g_llama_moe_layer_perf.add_branch_expert_locked(layer, (uint32_t) id, hot);
            }
        }
    }
}

bool llama_moe_layer_perf_eval_callback(ggml_tensor * t, bool ask, void * user_data) {
    const auto * ctx = static_cast<const llama_context *>(user_data);
    const llama_moe_layer_perf_mode mode = llama_moe_layer_perf_get_mode(ctx);

    if (mode == LLAMA_MOE_LAYER_PERF_MODE_OFF) {
        return true;
    }

    if (t == nullptr || t->name == nullptr || t->name[0] == '\0') {
        return true;
    }

    const char * name = t->name;

    if (ask) {
        return mode == LLAMA_MOE_LAYER_PERF_MODE_FULL ?
            llama_moe_layer_perf_node_classifier::is_any_node(name) :
            llama_moe_layer_perf_node_classifier::is_update_node(name);
    }

    if (!llama_moe_layer_perf_node_classifier::is_any_node(name)) {
        return true;
    }

    const int layer = llama_moe_layer_perf_node_classifier::parse_layer_from_name(name);
    if (layer < 0) {
        return true;
    }

    std::lock_guard<std::mutex> lock(g_llama_moe_layer_perf.mutex);

    if (!g_llama_moe_layer_perf.active) {
        return true;
    }

    if ((uint32_t) layer >= g_llama_moe_layer_perf.layers.size()) {
        return true;
    }

    uint64_t elapsed_us = 0;

    auto & dst = g_llama_moe_layer_perf.layers[layer];

    if (mode == LLAMA_MOE_LAYER_PERF_MODE_FULL) {
        const int64_t now_us = ggml_time_us();

        if (g_llama_moe_layer_perf.last_callback_us != 0 && now_us > g_llama_moe_layer_perf.last_callback_us) {
            elapsed_us = (uint64_t) (now_us - g_llama_moe_layer_perf.last_callback_us);
        }

        g_llama_moe_layer_perf.last_callback_us = now_us;
    }

    if (mode == LLAMA_MOE_LAYER_PERF_MODE_FULL && elapsed_us > 0) {
        g_llama_moe_layer_perf.add_locked(dst.total_moe_time_us, elapsed_us);

        if (llama_moe_layer_perf_node_classifier::is_expert_matmul_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.expert_matmul_time_us, elapsed_us);
        }

        if (llama_moe_layer_perf_node_classifier::is_hot_branch_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.hot_branch_time_us, elapsed_us);
        } else if (llama_moe_layer_perf_node_classifier::is_cold_branch_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.cold_branch_time_us, elapsed_us);
        }

        if (llama_moe_layer_perf_node_classifier::is_hot_expert_matmul_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.hot_expert_matmul_time_us, elapsed_us);
        } else if (llama_moe_layer_perf_node_classifier::is_cold_expert_matmul_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.cold_expert_matmul_time_us, elapsed_us);
        }

        if (llama_moe_layer_perf_node_classifier::is_worklist_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.worklist_time_us, elapsed_us);
        } else if (llama_moe_layer_perf_node_classifier::is_routing_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.routing_time_us, elapsed_us);
        } else if (llama_moe_layer_perf_node_classifier::is_merge_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.merge_time_us, elapsed_us);
        }

        if (llama_moe_layer_perf_node_classifier::is_hot_gather_scatter_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.hot_gather_scatter_time_us, elapsed_us);
        } else if (llama_moe_layer_perf_node_classifier::is_cold_gather_scatter_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.cold_gather_scatter_time_us, elapsed_us);
        }

        if (llama_moe_layer_perf_node_classifier::is_gate_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.gate_time_us, elapsed_us);
        } else if (llama_moe_layer_perf_node_classifier::is_up_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.up_time_us, elapsed_us);
        } else if (llama_moe_layer_perf_node_classifier::is_down_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.down_time_us, elapsed_us);
        }
    }

    if (llama_moe_layer_perf_node_classifier::is_topk_node(name)) {
        llama_moe_layer_perf_count_topk_locked((uint32_t) layer, t);
    } else if (llama_moe_layer_perf_node_classifier::is_hot_count_node(name)) {
        llama_moe_layer_perf_count_worklist_count_locked((uint32_t) layer, t, true);
    } else if (llama_moe_layer_perf_node_classifier::is_cold_count_node(name)) {
        llama_moe_layer_perf_count_worklist_count_locked((uint32_t) layer, t, false);
    } else if (llama_moe_layer_perf_node_classifier::is_hot_expert_ids_node(name)) {
        llama_moe_layer_perf_count_branch_experts_locked((uint32_t) layer, t, true);
    } else if (llama_moe_layer_perf_node_classifier::is_cold_ids_node(name)) {
        llama_moe_layer_perf_count_branch_experts_locked((uint32_t) layer, t, false);
    }

    return true;
}

void llama_moe_layer_perf_collect_parallel_metrics(ggml_backend_sched_t sched) {
    if (sched == nullptr) {
        return;
    }

    const llama_moe_layer_perf_mode mode = llama_moe_layer_perf_get_mode(nullptr);
    if (mode == LLAMA_MOE_LAYER_PERF_MODE_OFF) {
        return;
    }
    const bool full_mode = mode == LLAMA_MOE_LAYER_PERF_MODE_FULL;

    const int n_metrics = ggml_backend_sched_get_moe_hot_cache_parallel_perf(sched, nullptr, 0);
    if (n_metrics <= 0) {
        return;
    }

    std::vector<ggml_backend_sched_moe_hot_cache_parallel_perf> metrics(n_metrics);
    ggml_backend_sched_get_moe_hot_cache_parallel_perf(sched, metrics.data(), n_metrics);

    std::lock_guard<std::mutex> lock(g_llama_moe_layer_perf.mutex);

    if (!g_llama_moe_layer_perf.active) {
        return;
    }

    for (const auto & metric : metrics) {
        if (metric.layer < 0 || (uint32_t) metric.layer >= g_llama_moe_layer_perf.layers.size()) {
            continue;
        }

        auto & dst = g_llama_moe_layer_perf.layers[metric.layer];
        const bool slots_counted_by_callback =
            dst.hot_worklist_calls > dst.parallel_regions ||
            dst.cold_worklist_calls > dst.parallel_regions;

        if (!slots_counted_by_callback) {
            g_llama_moe_layer_perf.add_locked(dst.calls, 1);
            g_llama_moe_layer_perf.add_locked(dst.hot_worklist_calls, 1);
            g_llama_moe_layer_perf.add_locked(dst.cold_worklist_calls, 1);
            g_llama_moe_layer_perf.add_locked(dst.hot_slots_total, metric.parallel_hot_count);
            g_llama_moe_layer_perf.add_locked(dst.cold_slots_total, metric.parallel_cold_count);
            if (metric.parallel_hot_count == 0) {
                g_llama_moe_layer_perf.add_locked(dst.hot_zero_calls, 1);
            }
            if (metric.parallel_cold_count == 0) {
                g_llama_moe_layer_perf.add_locked(dst.cold_zero_calls, 1);
            }
        }
        g_llama_moe_layer_perf.add_locked(dst.parallel_regions, 1);
        g_llama_moe_layer_perf.add_locked(dst.parallel_hot_lane_wall_time_us, metric.parallel_hot_lane_wall_time_us);
        g_llama_moe_layer_perf.add_locked(dst.parallel_cold_lane_wall_time_us, metric.parallel_cold_lane_wall_time_us);
        g_llama_moe_layer_perf.add_locked(dst.parallel_join_wait_time_us, metric.parallel_join_wait_time_us);

        if (full_mode) {
            g_llama_moe_layer_perf.add_locked(dst.parallel_region_wall_time_us, metric.parallel_region_wall_time_us);
            g_llama_moe_layer_perf.add_locked(dst.parallel_overlap_estimate_us, metric.parallel_overlap_estimate_us);
            g_llama_moe_layer_perf.add_locked(dst.parallel_hot_launches, metric.parallel_hot_launches);
            g_llama_moe_layer_perf.add_locked(dst.parallel_cold_launches, metric.parallel_cold_launches);
            g_llama_moe_layer_perf.add_locked(dst.parallel_hot_skips_zero, metric.parallel_hot_skips_zero);
            g_llama_moe_layer_perf.add_locked(dst.parallel_cold_skips_zero, metric.parallel_cold_skips_zero);
            g_llama_moe_layer_perf.add_locked(dst.parallel_fallbacks, metric.parallel_fallbacks);
            g_llama_moe_layer_perf.add_locked(dst.parallel_fallback_incomplete, metric.parallel_fallback_incomplete);
            g_llama_moe_layer_perf.add_locked(dst.parallel_fallback_count_not_prefix, metric.parallel_fallback_count_not_prefix);
            g_llama_moe_layer_perf.add_locked(dst.parallel_fallback_bad_split_order, metric.parallel_fallback_bad_split_order);
            g_llama_moe_layer_perf.add_locked(dst.parallel_fallback_same_backend, metric.parallel_fallback_same_backend);
            g_llama_moe_layer_perf.add_locked(dst.parallel_fallback_hot_spans_backends, metric.parallel_fallback_hot_spans_backends);
            g_llama_moe_layer_perf.add_locked(dst.parallel_fallback_cold_spans_backends, metric.parallel_fallback_cold_spans_backends);
            g_llama_moe_layer_perf.add_locked(dst.parallel_fallback_hot_not_cuda, metric.parallel_fallback_hot_not_cuda);
            g_llama_moe_layer_perf.add_locked(dst.parallel_fallback_cold_not_cpu, metric.parallel_fallback_cold_not_cpu);
            g_llama_moe_layer_perf.add_locked(dst.parallel_fallback_count_readback, metric.parallel_fallback_count_readback);
            g_llama_moe_layer_perf.add_locked(dst.parallel_fallback_threshold, metric.parallel_fallback_threshold);
            g_llama_moe_layer_perf.add_locked(dst.parallel_fallback_zero_output, metric.parallel_fallback_zero_output);
            g_llama_moe_layer_perf.add_locked(dst.parallel_fallback_other, metric.parallel_fallback_other);
            if (metric.parallel_split_debug_samples > 0) {
                g_llama_moe_layer_perf.add_locked(dst.parallel_split_debug_samples, metric.parallel_split_debug_samples);
                dst.parallel_split_debug_hot_begin = metric.parallel_split_debug_hot_begin;
                dst.parallel_split_debug_hot_end = metric.parallel_split_debug_hot_end;
                dst.parallel_split_debug_cold_begin = metric.parallel_split_debug_cold_begin;
                dst.parallel_split_debug_cold_end = metric.parallel_split_debug_cold_end;
                dst.parallel_split_debug_join = metric.parallel_split_debug_join;
                dst.parallel_split_debug_hot_count = metric.parallel_split_debug_hot_count;
                dst.parallel_split_debug_cold_count = metric.parallel_split_debug_cold_count;
                dst.parallel_split_debug_hot_backend = metric.parallel_split_debug_hot_backend;
                dst.parallel_split_debug_cold_backend = metric.parallel_split_debug_cold_backend;
                dst.parallel_split_debug_join_backend = metric.parallel_split_debug_join_backend;
            }
        }
    }
}

bool llama_moe_layer_perf_graph_compute_begin(llama_context * ctx, ggml_backend_sched_t sched) {
    const auto & cparams = ctx->get_cparams();
    const auto & model = ctx->get_model();
    const bool enabled = !cparams.warmup && llama_moe_layer_perf_is_enabled(ctx) && model.hparams.n_expert > 0;

    ggml_backend_sched_set_moe_hot_cache_parallel_perf_enabled(sched, enabled);
    if (!enabled) {
        return false;
    }

    ggml_backend_sched_set_eval_callback(sched, llama_moe_layer_perf_eval_callback, ctx);
    llama_moe_layer_perf_begin(model.hparams.n_layer, model.hparams.n_expert, model.hparams.n_expert_used);

    return true;
}

void llama_moe_layer_perf_graph_compute_end(llama_context * ctx, ggml_backend_sched_t sched) {
    const auto & cparams = ctx->get_cparams();

    llama_moe_layer_perf_collect_parallel_metrics(sched);
    llama_moe_layer_perf_end();
    ggml_backend_sched_set_eval_callback(sched, cparams.cb_eval, cparams.cb_eval_user_data);
}

static llama_moe_layer_perf_json_layer_snapshot llama_moe_layer_perf_make_json_layer_snapshot(
        const llama_moe_layer_perf_layer & src) {
    llama_moe_layer_perf_json_layer_snapshot dst;

    dst.calls = src.calls;
    dst.expert_hits_total = src.expert_hits_total;
    dst.hot_slots_total = src.hot_slots_total;
    dst.cold_slots_total = src.cold_slots_total;
    dst.hot_worklist_calls = src.hot_worklist_calls;
    dst.cold_worklist_calls = src.cold_worklist_calls;

    dst.total_moe_time_us = src.total_moe_time_us;
    dst.hot_branch_time_us = src.hot_branch_time_us;
    dst.cold_branch_time_us = src.cold_branch_time_us;
    dst.hot_expert_matmul_time_us = src.hot_expert_matmul_time_us;
    dst.cold_expert_matmul_time_us = src.cold_expert_matmul_time_us;
    dst.worklist_time_us = src.worklist_time_us;
    dst.routing_time_us = src.routing_time_us;
    dst.merge_time_us = src.merge_time_us;
    dst.hot_gather_scatter_time_us = src.hot_gather_scatter_time_us;
    dst.cold_gather_scatter_time_us = src.cold_gather_scatter_time_us;

    dst.parallel_region_wall_time_us = src.parallel_region_wall_time_us;
    dst.parallel_hot_lane_wall_time_us = src.parallel_hot_lane_wall_time_us;
    dst.parallel_cold_lane_wall_time_us = src.parallel_cold_lane_wall_time_us;
    dst.parallel_join_wait_time_us = src.parallel_join_wait_time_us;
    dst.parallel_overlap_estimate_us = src.parallel_overlap_estimate_us;
    dst.parallel_hot_launches = src.parallel_hot_launches;
    dst.parallel_cold_launches = src.parallel_cold_launches;
    dst.parallel_hot_skips_zero = src.parallel_hot_skips_zero;
    dst.parallel_cold_skips_zero = src.parallel_cold_skips_zero;
    dst.parallel_fallbacks = src.parallel_fallbacks;
    dst.parallel_fallback_incomplete = src.parallel_fallback_incomplete;
    dst.parallel_fallback_count_not_prefix = src.parallel_fallback_count_not_prefix;
    dst.parallel_fallback_bad_split_order = src.parallel_fallback_bad_split_order;
    dst.parallel_fallback_same_backend = src.parallel_fallback_same_backend;
    dst.parallel_fallback_hot_spans_backends = src.parallel_fallback_hot_spans_backends;
    dst.parallel_fallback_cold_spans_backends = src.parallel_fallback_cold_spans_backends;
    dst.parallel_fallback_hot_not_cuda = src.parallel_fallback_hot_not_cuda;
    dst.parallel_fallback_cold_not_cpu = src.parallel_fallback_cold_not_cpu;
    dst.parallel_fallback_count_readback = src.parallel_fallback_count_readback;
    dst.parallel_fallback_threshold = src.parallel_fallback_threshold;
    dst.parallel_fallback_zero_output = src.parallel_fallback_zero_output;
    dst.parallel_fallback_other = src.parallel_fallback_other;
    dst.parallel_split_debug_samples = src.parallel_split_debug_samples;
    dst.parallel_split_debug_hot_begin = src.parallel_split_debug_hot_begin;
    dst.parallel_split_debug_hot_end = src.parallel_split_debug_hot_end;
    dst.parallel_split_debug_cold_begin = src.parallel_split_debug_cold_begin;
    dst.parallel_split_debug_cold_end = src.parallel_split_debug_cold_end;
    dst.parallel_split_debug_join = src.parallel_split_debug_join;
    dst.parallel_split_debug_hot_count = src.parallel_split_debug_hot_count;
    dst.parallel_split_debug_cold_count = src.parallel_split_debug_cold_count;
    dst.parallel_split_debug_hot_backend = src.parallel_split_debug_hot_backend;
    dst.parallel_split_debug_cold_backend = src.parallel_split_debug_cold_backend;
    dst.parallel_split_debug_join_backend = src.parallel_split_debug_join_backend;

    dst.experts = src.experts;
    dst.hot_experts = src.hot_experts;
    dst.cold_experts = src.cold_experts;

    return dst;
}

static llama_moe_layer_perf_json_snapshot llama_moe_layer_perf_make_json_snapshot_locked(
        llama_moe_layer_perf_mode mode) {
    llama_moe_layer_perf_json_snapshot snapshot;
    snapshot.enabled = true;
    snapshot.mode = mode;
    snapshot.n_expert = g_llama_moe_layer_perf.n_expert;
    snapshot.n_expert_used = g_llama_moe_layer_perf.n_expert_used;
    snapshot.updates = g_llama_moe_layer_perf.updates;
    snapshot.overflow_resets = g_llama_moe_layer_perf.overflow_resets;
    snapshot.layers.reserve(g_llama_moe_layer_perf.layers.size());

    for (const auto & layer : g_llama_moe_layer_perf.layers) {
        snapshot.layers.push_back(llama_moe_layer_perf_make_json_layer_snapshot(layer));
    }

    return snapshot;
}

const char * llama_moe_layer_perf_json(struct llama_context * ctx) {
    static thread_local std::string result;

    const llama_moe_layer_perf_mode mode = llama_moe_layer_perf_get_mode(ctx);
    if (mode == LLAMA_MOE_LAYER_PERF_MODE_OFF) {
        result = llama_moe_layer_perf_json_serializer::serialize_disabled();
        return result.c_str();
    }

    llama_moe_layer_perf_json_snapshot snapshot;
    {
        std::lock_guard<std::mutex> lock(g_llama_moe_layer_perf.mutex);
        snapshot = llama_moe_layer_perf_make_json_snapshot_locked(mode);
    }

    result = llama_moe_layer_perf_json_serializer::serialize(snapshot);
    return result.c_str();
}
