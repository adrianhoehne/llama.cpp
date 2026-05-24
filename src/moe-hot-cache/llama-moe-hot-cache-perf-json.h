#pragma once

#include "llama-moe-hot-cache-perf.h"

#include <cstdint>
#include <string>
#include <vector>

struct llama_moe_layer_perf_json_layer_snapshot {
    uint64_t calls = 0;
    uint64_t expert_hits_total = 0;
    uint64_t hot_slots_total = 0;
    uint64_t cold_slots_total = 0;
    uint64_t hot_worklist_calls = 0;
    uint64_t cold_worklist_calls = 0;

    uint64_t total_moe_time_us = 0;
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

    std::vector<uint64_t> experts;
    std::vector<uint64_t> hot_experts;
    std::vector<uint64_t> cold_experts;
};

struct llama_moe_layer_perf_json_snapshot {
    bool enabled = true;
    llama_moe_layer_perf_mode mode = LLAMA_MOE_LAYER_PERF_MODE_FULL;
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint64_t updates = 0;
    uint64_t overflow_resets = 0;
    std::vector<llama_moe_layer_perf_json_layer_snapshot> layers;
};

class llama_moe_layer_perf_json_serializer {
public:
    static std::string serialize_disabled();
    static std::string serialize(const llama_moe_layer_perf_json_snapshot & snapshot);
};
