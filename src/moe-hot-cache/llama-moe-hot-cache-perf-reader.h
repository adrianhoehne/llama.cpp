#pragma once

#include "llama-moe-hot-cache-perf-state.h"

#include <cstdint>

struct ggml_tensor;

class llama_moe_layer_perf_tensor_reader {
public:
    static void count_topk_locked(llama_moe_layer_perf_state & state, uint32_t layer, ggml_tensor * t);
    static void count_worklist_count_locked(llama_moe_layer_perf_state & state, uint32_t layer, ggml_tensor * t, bool hot);
    static void count_branch_experts_locked(llama_moe_layer_perf_state & state, uint32_t layer, ggml_tensor * t, bool hot);
};
