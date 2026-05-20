#include "llama-moe-hot-cache-perf.h"

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
#include <sstream>
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

static int llama_moe_parse_layer_from_name(const char * name) {
    if (name == nullptr) {
        return -1;
    }

    const char * p = std::strstr(name, "ffn_moe_");
    if (p == nullptr) {
        return -1;
    }

    const char * dash = std::strchr(p, '-');
    if (dash == nullptr) {
        return -1;
    }

    dash++;

    if (*dash < '0' || *dash > '9') {
        return -1;
    }

    return std::atoi(dash);
}

static bool llama_moe_name_contains(const char * name, const char * needle) {
    return name != nullptr && std::strstr(name, needle) != nullptr;
}

static bool llama_moe_is_any_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_");
}

static bool llama_moe_is_update_node(const char * name);

static bool llama_moe_is_topk_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_topk-");
}

static bool llama_moe_is_hot_count_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_hot_count-");
}

static bool llama_moe_is_cold_count_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_cold_count-");
}

static bool llama_moe_is_hot_expert_ids_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_hot_expert_ids_compact-");
}

static bool llama_moe_is_cold_ids_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_cold_ids_compact-");
}

static bool llama_moe_is_update_node(const char * name) {
    return llama_moe_is_topk_node(name) ||
           llama_moe_is_hot_count_node(name) ||
           llama_moe_is_cold_count_node(name) ||
           llama_moe_is_hot_expert_ids_node(name) ||
           llama_moe_is_cold_ids_node(name);
}

static bool llama_moe_is_gate_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_gate-") ||
           llama_moe_name_contains(name, "ffn_moe_hot_gate-") ||
           llama_moe_name_contains(name, "ffn_moe_cold_gate-");
}

static bool llama_moe_is_up_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_up-") ||
           llama_moe_name_contains(name, "ffn_moe_hot_up-") ||
           llama_moe_name_contains(name, "ffn_moe_cold_up-");
}

static bool llama_moe_is_down_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_down-") ||
           llama_moe_name_contains(name, "ffn_moe_hot_down-") ||
           llama_moe_name_contains(name, "ffn_moe_cold_down-");
}

static bool llama_moe_is_hot_gate_up_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_hot_gate_up-");
}

static bool llama_moe_is_cold_gate_up_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_cold_gate_up-");
}

static bool llama_moe_is_hot_gate_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_hot_gate-");
}

static bool llama_moe_is_cold_gate_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_cold_gate-");
}

static bool llama_moe_is_hot_up_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_hot_up-");
}

static bool llama_moe_is_cold_up_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_cold_up-");
}

static bool llama_moe_is_hot_down_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_hot_down-");
}

static bool llama_moe_is_cold_down_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_cold_down-");
}

static bool llama_moe_is_hot_branch_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_hot_");
}

static bool llama_moe_is_cold_branch_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_cold_");
}

static bool llama_moe_is_worklist_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_worklist-");
}

static bool llama_moe_is_routing_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_logits-") ||
           llama_moe_name_contains(name, "ffn_moe_probs-") ||
           llama_moe_name_contains(name, "ffn_moe_argsort-") ||
           llama_moe_name_contains(name, "ffn_moe_topk-") ||
           llama_moe_name_contains(name, "ffn_moe_weights-") ||
           llama_moe_name_contains(name, "ffn_moe_weights_sum-") ||
           llama_moe_name_contains(name, "ffn_moe_weights_sum_clamped-") ||
           llama_moe_name_contains(name, "ffn_moe_weights_norm-") ||
           llama_moe_name_contains(name, "ffn_moe_weights_scaled-");
}

static bool llama_moe_is_merge_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_hot_cold_slots-") ||
           llama_moe_name_contains(name, "ffn_moe_out-");
}

static bool llama_moe_is_hot_gather_scatter_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_hot_ids_compact-") ||
           llama_moe_name_contains(name, "ffn_moe_hot_expert_ids_compact-") ||
           llama_moe_name_contains(name, "ffn_moe_hot_src_slots-") ||
           llama_moe_name_contains(name, "ffn_moe_hot_token_ids-") ||
           llama_moe_name_contains(name, "ffn_moe_hot_weights_compact-") ||
           llama_moe_name_contains(name, "ffn_moe_hot_inputs-") ||
           llama_moe_name_contains(name, "ffn_moe_hot_slots-");
}

static bool llama_moe_is_cold_gather_scatter_node(const char * name) {
    return llama_moe_name_contains(name, "ffn_moe_cold_ids_compact-") ||
           llama_moe_name_contains(name, "ffn_moe_cold_src_slots-") ||
           llama_moe_name_contains(name, "ffn_moe_cold_token_ids-") ||
           llama_moe_name_contains(name, "ffn_moe_cold_weights_compact-") ||
           llama_moe_name_contains(name, "ffn_moe_cold_inputs-") ||
           llama_moe_name_contains(name, "ffn_moe_cold_slots-");
}

static bool llama_moe_is_hot_expert_matmul_node(const char * name) {
    return llama_moe_is_hot_gate_up_node(name) ||
           llama_moe_is_hot_gate_node(name)    ||
           llama_moe_is_hot_up_node(name)      ||
           llama_moe_is_hot_down_node(name);
}

static bool llama_moe_is_cold_expert_matmul_node(const char * name) {
    return llama_moe_is_cold_gate_up_node(name) ||
           llama_moe_is_cold_gate_node(name)    ||
           llama_moe_is_cold_up_node(name)      ||
           llama_moe_is_cold_down_node(name);
}

static bool llama_moe_is_expert_matmul_node(const char * name) {
    return llama_moe_is_gate_node(name) ||
           llama_moe_is_up_node(name)   ||
           llama_moe_is_down_node(name);
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
            llama_moe_is_any_node(name) :
            llama_moe_is_update_node(name);
    }

    if (!llama_moe_is_any_node(name)) {
        return true;
    }

    const int layer = llama_moe_parse_layer_from_name(name);
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

        if (llama_moe_is_expert_matmul_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.expert_matmul_time_us, elapsed_us);
        }

        if (llama_moe_is_hot_branch_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.hot_branch_time_us, elapsed_us);
        } else if (llama_moe_is_cold_branch_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.cold_branch_time_us, elapsed_us);
        }

        if (llama_moe_is_hot_expert_matmul_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.hot_expert_matmul_time_us, elapsed_us);
        } else if (llama_moe_is_cold_expert_matmul_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.cold_expert_matmul_time_us, elapsed_us);
        }

        if (llama_moe_is_worklist_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.worklist_time_us, elapsed_us);
        } else if (llama_moe_is_routing_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.routing_time_us, elapsed_us);
        } else if (llama_moe_is_merge_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.merge_time_us, elapsed_us);
        }

        if (llama_moe_is_hot_gather_scatter_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.hot_gather_scatter_time_us, elapsed_us);
        } else if (llama_moe_is_cold_gather_scatter_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.cold_gather_scatter_time_us, elapsed_us);
        }

        if (llama_moe_is_gate_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.gate_time_us, elapsed_us);
        } else if (llama_moe_is_up_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.up_time_us, elapsed_us);
        } else if (llama_moe_is_down_node(name)) {
            g_llama_moe_layer_perf.add_locked(dst.down_time_us, elapsed_us);
        }
    }

    if (llama_moe_is_topk_node(name)) {
        llama_moe_layer_perf_count_topk_locked((uint32_t) layer, t);
    } else if (llama_moe_is_hot_count_node(name)) {
        llama_moe_layer_perf_count_worklist_count_locked((uint32_t) layer, t, true);
    } else if (llama_moe_is_cold_count_node(name)) {
        llama_moe_layer_perf_count_worklist_count_locked((uint32_t) layer, t, false);
    } else if (llama_moe_is_hot_expert_ids_node(name)) {
        llama_moe_layer_perf_count_branch_experts_locked((uint32_t) layer, t, true);
    } else if (llama_moe_is_cold_ids_node(name)) {
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

const char * llama_moe_layer_perf_json(struct llama_context * ctx) {
    static thread_local std::string result;

    const llama_moe_layer_perf_mode mode = llama_moe_layer_perf_get_mode(ctx);
    if (mode == LLAMA_MOE_LAYER_PERF_MODE_OFF) {
        result = "{\"enabled\":false,\"mode\":\"off\",\"schema\":\"llama.cpp.moe_layer_opt_perf.v1\",\"layers\":[]}";
        return result.c_str();
    }
    const bool full_mode = mode == LLAMA_MOE_LAYER_PERF_MODE_FULL;

    std::lock_guard<std::mutex> lock(g_llama_moe_layer_perf.mutex);

    std::ostringstream out;

    const auto non_empty_layer = [](const llama_moe_layer_perf_layer & layer) {
        return layer.calls != 0 ||
               layer.expert_hits_total != 0 ||
               layer.total_moe_time_us != 0 ||
               layer.parallel_fallbacks != 0;
    };

    const auto per_call = [](uint64_t value, uint64_t calls) {
        return calls > 0 ? (double) value / (double) calls : 0.0;
    };

    const auto ratio = [](uint64_t value, uint64_t total) {
        return total > 0 ? (double) value / (double) total : 0.0;
    };

    const auto has_expert_counts = [](const std::vector<uint64_t> & counts) {
        return std::any_of(counts.begin(), counts.end(), [](uint64_t count) {
            return count != 0;
        });
    };

    const auto raw_counts_are_cold = [&](const llama_moe_layer_perf_layer & layer) {
        return layer.expert_hits_total != 0 &&
               !has_expert_counts(layer.hot_experts) &&
               !has_expert_counts(layer.cold_experts);
    };

    uint64_t summary_calls = 0;
    uint64_t summary_hot_slots = 0;
    uint64_t summary_cold_slots = 0;
    uint64_t summary_total_moe_time_us = 0;
    uint64_t summary_routing_time_us = 0;
    uint64_t summary_worklist_time_us = 0;
    uint64_t summary_merge_time_us = 0;
    uint64_t summary_hot_branch_time_us = 0;
    uint64_t summary_cold_branch_time_us = 0;
    uint64_t summary_hot_expert_matmul_time_us = 0;
    uint64_t summary_cold_expert_matmul_time_us = 0;
    uint64_t summary_hot_gather_scatter_time_us = 0;
    uint64_t summary_cold_gather_scatter_time_us = 0;
    uint64_t summary_parallel_region_wall_time_us = 0;
    uint64_t summary_parallel_hot_lane_wall_time_us = 0;
    uint64_t summary_parallel_cold_lane_wall_time_us = 0;
    uint64_t summary_parallel_join_wait_time_us = 0;
    uint64_t summary_parallel_overlap_estimate_us = 0;
    uint64_t summary_parallel_hot_launches = 0;
    uint64_t summary_parallel_cold_launches = 0;
    uint64_t summary_parallel_fallbacks = 0;

    for (const auto & layer : g_llama_moe_layer_perf.layers) {
        if (!non_empty_layer(layer)) {
            continue;
        }

        const bool raw_as_cold = raw_counts_are_cold(layer);

        summary_calls += layer.calls;
        summary_hot_slots += raw_as_cold ? 0 : layer.hot_slots_total;
        summary_cold_slots += raw_as_cold ? layer.expert_hits_total : layer.cold_slots_total;
        summary_total_moe_time_us += layer.total_moe_time_us;
        summary_routing_time_us += layer.routing_time_us;
        summary_worklist_time_us += layer.worklist_time_us;
        summary_merge_time_us += layer.merge_time_us;
        summary_hot_branch_time_us += layer.hot_branch_time_us;
        summary_cold_branch_time_us += layer.cold_branch_time_us;
        summary_hot_expert_matmul_time_us += layer.hot_expert_matmul_time_us;
        summary_cold_expert_matmul_time_us += layer.cold_expert_matmul_time_us;
        summary_hot_gather_scatter_time_us += layer.hot_gather_scatter_time_us;
        summary_cold_gather_scatter_time_us += layer.cold_gather_scatter_time_us;
        summary_parallel_region_wall_time_us += layer.parallel_region_wall_time_us;
        summary_parallel_hot_lane_wall_time_us += layer.parallel_hot_lane_wall_time_us;
        summary_parallel_cold_lane_wall_time_us += layer.parallel_cold_lane_wall_time_us;
        summary_parallel_join_wait_time_us += layer.parallel_join_wait_time_us;
        summary_parallel_overlap_estimate_us += layer.parallel_overlap_estimate_us;
        summary_parallel_hot_launches += layer.parallel_hot_launches;
        summary_parallel_cold_launches += layer.parallel_cold_launches;
        summary_parallel_fallbacks += layer.parallel_fallbacks;
    }

    const uint64_t summary_slots = summary_hot_slots + summary_cold_slots;

    out << "{";
    out << "\"enabled\":true,";
    out << "\"mode\":\"" << llama_moe_layer_perf_mode_name(mode) << "\",";
    out << "\"schema\":\"llama.cpp.moe_layer_opt_perf.v1\",";
    out << "\"n_expert\":" << g_llama_moe_layer_perf.n_expert << ",";
    out << "\"n_expert_used\":" << g_llama_moe_layer_perf.n_expert_used << ",";
    out << "\"updates\":" << g_llama_moe_layer_perf.updates << ",";
    out << "\"overflow_resets\":" << g_llama_moe_layer_perf.overflow_resets << ",";
    out << "\"summary\":{";
    out << "\"layer_calls\":" << summary_calls;
    out << ",\"hot_slot_ratio\":" << ratio(summary_hot_slots, summary_slots);
    out << ",\"parallel_hot_lane_wall_time_per_call_us\":" << per_call(summary_parallel_hot_lane_wall_time_us, summary_calls);
    out << ",\"parallel_cold_lane_wall_time_per_call_us\":" << per_call(summary_parallel_cold_lane_wall_time_us, summary_calls);
    out << ",\"parallel_join_wait_time_per_call_us\":" << per_call(summary_parallel_join_wait_time_us, summary_calls);
    if (full_mode) {
        out << ",\"total_moe_time_per_call_us\":" << per_call(summary_total_moe_time_us, summary_calls);
        out << ",\"routing_time_per_call_us\":" << per_call(summary_routing_time_us, summary_calls);
        out << ",\"worklist_time_per_call_us\":" << per_call(summary_worklist_time_us, summary_calls);
        out << ",\"merge_time_per_call_us\":" << per_call(summary_merge_time_us, summary_calls);
        out << ",\"hot_branch_time_per_call_us\":" << per_call(summary_hot_branch_time_us, summary_calls);
        out << ",\"cold_branch_time_per_call_us\":" << per_call(summary_cold_branch_time_us, summary_calls);
        out << ",\"hot_expert_matmul_time_per_call_us\":" << per_call(summary_hot_expert_matmul_time_us, summary_calls);
        out << ",\"cold_expert_matmul_time_per_call_us\":" << per_call(summary_cold_expert_matmul_time_us, summary_calls);
        out << ",\"hot_gather_scatter_time_per_call_us\":" << per_call(summary_hot_gather_scatter_time_us, summary_calls);
        out << ",\"cold_gather_scatter_time_per_call_us\":" << per_call(summary_cold_gather_scatter_time_us, summary_calls);
        out << ",\"parallel_region_wall_time_per_call_us\":" << per_call(summary_parallel_region_wall_time_us, summary_calls);
        out << ",\"parallel_overlap_estimate_per_call_us\":" << per_call(summary_parallel_overlap_estimate_us, summary_calls);
        out << ",\"parallel_hot_launches\":" << summary_parallel_hot_launches;
        out << ",\"parallel_cold_launches\":" << summary_parallel_cold_launches;
        out << ",\"parallel_fallbacks\":" << summary_parallel_fallbacks;
    }
    out << "},";
    out << "\"layers\":[";

    const auto write_expert_counts = [&](const std::vector<uint64_t> & counts) {
        out << "[";

        bool first_expert = true;

        for (size_t ex = 0; ex < counts.size(); ++ex) {
            if (counts[ex] == 0) {
                continue;
            }

            if (!first_expert) {
                out << ",";
            }

            first_expert = false;

            out << "[" << ex << "," << counts[ex] << "]";
        }

        out << "]";
    };

    const auto write_fallback_reason = [&](bool & first, const char * name, uint64_t value) {
        if (value == 0) {
            return;
        }

        if (!first) {
            out << ",";
        }

        first = false;
        out << "\"" << name << "\":" << value;
    };

    bool first_layer = true;

    for (size_t il = 0; il < g_llama_moe_layer_perf.layers.size(); ++il) {
        const auto & layer = g_llama_moe_layer_perf.layers[il];

        if (!non_empty_layer(layer)) {
            continue;
        }

        if (!first_layer) {
            out << ",";
        }

        first_layer = false;

        const bool raw_as_cold = raw_counts_are_cold(layer);
        const uint64_t hot_slots_total = raw_as_cold ? 0 : layer.hot_slots_total;
        const uint64_t cold_slots_total = raw_as_cold ? layer.expert_hits_total : layer.cold_slots_total;

        const double hot_slots_per_call =
            raw_as_cold ? 0.0 :
            layer.hot_worklist_calls > 0 ? (double) hot_slots_total / (double) layer.hot_worklist_calls : per_call(hot_slots_total, layer.calls);

        const double cold_slots_per_call =
            raw_as_cold ? per_call(cold_slots_total, layer.calls) :
            layer.cold_worklist_calls > 0 ? (double) cold_slots_total / (double) layer.cold_worklist_calls : per_call(cold_slots_total, layer.calls);

        const uint64_t slots_total = hot_slots_total + cold_slots_total;

        out << "{";
        out << "\"layer\":" << il << ",";
        out << "\"calls\":" << layer.calls << ",";
        out << "\"hot_slots_total\":" << hot_slots_total << ",";
        out << "\"cold_slots_total\":" << cold_slots_total << ",";
        out << "\"hot_slots_per_call\":" << hot_slots_per_call << ",";
        out << "\"cold_slots_per_call\":" << cold_slots_per_call << ",";
        out << "\"hot_slot_ratio\":" << ratio(hot_slots_total, slots_total);
        out << ",\"parallel_hot_lane_wall_time_per_call_us\":" << per_call(layer.parallel_hot_lane_wall_time_us, layer.calls);
        out << ",\"parallel_cold_lane_wall_time_per_call_us\":" << per_call(layer.parallel_cold_lane_wall_time_us, layer.calls);
        out << ",\"parallel_join_wait_time_per_call_us\":" << per_call(layer.parallel_join_wait_time_us, layer.calls);

        if (full_mode) {
            out << ",\"total_moe_time_per_call_us\":" << per_call(layer.total_moe_time_us, layer.calls);
            out << ",\"routing_time_per_call_us\":" << per_call(layer.routing_time_us, layer.calls);
            out << ",\"worklist_time_per_call_us\":" << per_call(layer.worklist_time_us, layer.calls);
            out << ",\"merge_time_per_call_us\":" << per_call(layer.merge_time_us, layer.calls);
            out << ",\"hot_branch_time_per_call_us\":" << per_call(layer.hot_branch_time_us, layer.calls);
            out << ",\"cold_branch_time_per_call_us\":" << per_call(layer.cold_branch_time_us, layer.calls);
            out << ",\"hot_expert_matmul_time_per_call_us\":" << per_call(layer.hot_expert_matmul_time_us, layer.calls);
            out << ",\"cold_expert_matmul_time_per_call_us\":" << per_call(layer.cold_expert_matmul_time_us, layer.calls);
            out << ",\"hot_gather_scatter_time_per_call_us\":" << per_call(layer.hot_gather_scatter_time_us, layer.calls);
            out << ",\"cold_gather_scatter_time_per_call_us\":" << per_call(layer.cold_gather_scatter_time_us, layer.calls);
            out << ",\"parallel_region_wall_time_per_call_us\":" << per_call(layer.parallel_region_wall_time_us, layer.calls);
            out << ",\"parallel_overlap_estimate_per_call_us\":" << per_call(layer.parallel_overlap_estimate_us, layer.calls);
            out << ",\"parallel_hot_launches\":" << layer.parallel_hot_launches;
            out << ",\"parallel_cold_launches\":" << layer.parallel_cold_launches;
            out << ",\"parallel_hot_skips_zero\":" << layer.parallel_hot_skips_zero;
            out << ",\"parallel_cold_skips_zero\":" << layer.parallel_cold_skips_zero;
            out << ",\"parallel_fallbacks\":" << layer.parallel_fallbacks;

            if (layer.parallel_fallbacks > 0) {
                bool first_reason = true;
                out << ",\"parallel_fallback_reasons\":{";
                write_fallback_reason(first_reason, "incomplete", layer.parallel_fallback_incomplete);
                write_fallback_reason(first_reason, "count_not_prefix", layer.parallel_fallback_count_not_prefix);
                write_fallback_reason(first_reason, "bad_split_order", layer.parallel_fallback_bad_split_order);
                write_fallback_reason(first_reason, "same_backend", layer.parallel_fallback_same_backend);
                write_fallback_reason(first_reason, "hot_spans_backends", layer.parallel_fallback_hot_spans_backends);
                write_fallback_reason(first_reason, "cold_spans_backends", layer.parallel_fallback_cold_spans_backends);
                write_fallback_reason(first_reason, "hot_not_cuda", layer.parallel_fallback_hot_not_cuda);
                write_fallback_reason(first_reason, "cold_not_cpu", layer.parallel_fallback_cold_not_cpu);
                write_fallback_reason(first_reason, "count_readback", layer.parallel_fallback_count_readback);
                write_fallback_reason(first_reason, "threshold", layer.parallel_fallback_threshold);
                write_fallback_reason(first_reason, "zero_output", layer.parallel_fallback_zero_output);
                write_fallback_reason(first_reason, "other", layer.parallel_fallback_other);
                out << "}";
            }

            if (layer.parallel_split_debug_samples > 0) {
                out << ",\"parallel_split_debug\":{";
                out << "\"samples\":" << layer.parallel_split_debug_samples << ",";
                out << "\"last\":{";
                out << "\"hot\":[" << layer.parallel_split_debug_hot_begin << "," << layer.parallel_split_debug_hot_end << "],";
                out << "\"cold\":[" << layer.parallel_split_debug_cold_begin << "," << layer.parallel_split_debug_cold_end << "],";
                out << "\"join\":" << layer.parallel_split_debug_join << ",";
                out << "\"counts\":{\"hot\":" << layer.parallel_split_debug_hot_count << ",\"cold\":" << layer.parallel_split_debug_cold_count << "},";
                out << "\"backends\":{\"hot\":" << layer.parallel_split_debug_hot_backend << ",\"cold\":" << layer.parallel_split_debug_cold_backend << ",\"join\":" << layer.parallel_split_debug_join_backend << "}";
                out << "}}";
            }
        }

        out << ",\"experts\":";
        write_expert_counts(layer.experts);
        out << ",\"hot_experts\":";
        write_expert_counts(layer.hot_experts);
        out << ",\"cold_experts\":";
        write_expert_counts(raw_as_cold ? layer.experts : layer.cold_experts);

        out << "}";
    }

    out << "]}";

    result = out.str();
    return result.c_str();
}
