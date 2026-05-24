#pragma once

#include <cstdint>
#include <mutex>
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

struct llama_moe_layer_perf_state {
    std::mutex mutex;

    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;

    uint64_t updates = 0;
    uint64_t overflow_resets = 0;

    bool active = false;
    int64_t last_callback_us = 0;

    std::vector<llama_moe_layer_perf_layer> layers;

    void ensure_shape_locked(uint32_t n_layer, uint32_t experts, uint32_t used);
    void reset_locked(bool count_overflow = true);
    void begin_locked(uint32_t n_layer, uint32_t experts, uint32_t used);
    void end_locked();
    bool has_data_locked() const;
    void add_locked(uint64_t & dst, uint64_t add);
    void add_expert_locked(uint32_t layer, uint32_t expert);
    void add_branch_expert_locked(uint32_t layer, uint32_t expert, bool hot);
};
