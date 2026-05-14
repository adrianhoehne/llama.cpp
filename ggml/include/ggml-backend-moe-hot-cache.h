#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    struct ggml_backend_sched_moe_hot_cache_parallel_perf {
        int32_t  layer;
        uint64_t parallel_hot_count;
        uint64_t parallel_cold_count;
        uint64_t parallel_region_wall_time_us;
        uint64_t parallel_hot_lane_wall_time_us;
        uint64_t parallel_cold_lane_wall_time_us;
        uint64_t parallel_join_wait_time_us;
        uint64_t parallel_overlap_estimate_us;
        uint64_t parallel_hot_launches;
        uint64_t parallel_cold_launches;
        uint64_t parallel_hot_skips_zero;
        uint64_t parallel_cold_skips_zero;
        uint64_t parallel_fallbacks;
        uint64_t parallel_fallback_incomplete;
        uint64_t parallel_fallback_count_not_prefix;
        uint64_t parallel_fallback_bad_split_order;
        uint64_t parallel_fallback_same_backend;
        uint64_t parallel_fallback_hot_spans_backends;
        uint64_t parallel_fallback_cold_spans_backends;
        uint64_t parallel_fallback_hot_not_cuda;
        uint64_t parallel_fallback_cold_not_cpu;
        uint64_t parallel_fallback_count_readback;
        uint64_t parallel_fallback_threshold;
        uint64_t parallel_fallback_zero_output;
        uint64_t parallel_fallback_other;
    };

    GGML_API void ggml_backend_sched_set_moe_hot_cache_parallel_perf_enabled(
            ggml_backend_sched_t sched,
            bool enabled);

    // Experimental: annotate one Qwen3.5 MoE hot-cache fork/join region.
    // mode: 0 = off, 1 = auto, 2 = force.
    GGML_API void ggml_backend_sched_moe_hot_cache_parallel_region(
            ggml_backend_sched_t sched,
            int32_t layer,
            int mode,
            int64_t max_slots,
            struct ggml_tensor * hot_count,
            struct ggml_tensor * cold_count,
            struct ggml_tensor * hot_start,
            struct ggml_tensor * hot_end,
            struct ggml_tensor * hot_output,
            struct ggml_tensor * cold_start,
            struct ggml_tensor * cold_end,
            struct ggml_tensor * cold_output,
            struct ggml_tensor * join);

    GGML_API int ggml_backend_sched_get_moe_hot_cache_parallel_perf(
            ggml_backend_sched_t sched,
            struct ggml_backend_sched_moe_hot_cache_parallel_perf * out,
            int max_entries);

#ifdef  __cplusplus
}
#endif
