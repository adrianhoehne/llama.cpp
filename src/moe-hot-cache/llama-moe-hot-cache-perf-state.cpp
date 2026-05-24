#include "llama-moe-hot-cache-perf-state.h"

#include <algorithm>
#include <limits>

void llama_moe_layer_perf_state::ensure_shape_locked(uint32_t n_layer, uint32_t experts, uint32_t used) {
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

void llama_moe_layer_perf_state::reset_locked(bool count_overflow) {
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

void llama_moe_layer_perf_state::begin_locked(uint32_t n_layer, uint32_t experts, uint32_t used) {
    if (n_layer == 0 || experts == 0 || used == 0) {
        active = false;
        return;
    }

    ensure_shape_locked(n_layer, experts, used);
    active = true;
    last_callback_us = 0;

    if (updates == std::numeric_limits<uint64_t>::max()) {
        reset_locked();
    }

    updates++;
}

void llama_moe_layer_perf_state::end_locked() {
    active = false;
    last_callback_us = 0;
}

bool llama_moe_layer_perf_state::has_data_locked() const {
    return std::any_of(
            layers.begin(),
            layers.end(),
            [](const llama_moe_layer_perf_layer & layer) {
                return layer.calls != 0 ||
                       layer.expert_hits_total != 0 ||
                       layer.total_moe_time_us != 0 ||
                       layer.parallel_fallbacks != 0;
            });
}

void llama_moe_layer_perf_state::add_locked(uint64_t & dst, uint64_t add) {
    if (dst > std::numeric_limits<uint64_t>::max() - add) {
        reset_locked();
    }

    dst += add;
}

void llama_moe_layer_perf_state::add_expert_locked(uint32_t layer, uint32_t expert) {
    if (layer >= layers.size()) {
        return;
    }

    if (expert >= layers[layer].experts.size()) {
        return;
    }

    add_locked(layers[layer].experts[expert], 1);
    add_locked(layers[layer].expert_hits_total, 1);
}

void llama_moe_layer_perf_state::add_branch_expert_locked(uint32_t layer, uint32_t expert, bool hot) {
    if (layer >= layers.size()) {
        return;
    }

    auto & counts = hot ? layers[layer].hot_experts : layers[layer].cold_experts;
    if (expert >= counts.size()) {
        return;
    }

    add_locked(counts[expert], 1);
}
