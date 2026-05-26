#pragma once

#include "llama-moe-hot-cache-adapter.h"
#include "llama-moe-hot-cache-worklist.h"

#include <cstdint>

enum class llama_moe_hot_cache_graph_phase {
    warmup,
    decode,
    prompt_processing,
};

struct llama_moe_hot_cache_pp_execution_plan {
    llama_moe_hot_cache_graph_phase phase = llama_moe_hot_cache_graph_phase::decode;
    llama_moe_hot_cache_worklist_order worklist_order = llama_moe_hot_cache_worklist_order::token_major;
    bool branch_reduce_merge = false;
    bool compact_cold_reduce = false;

    bool is_prompt_processing() const {
        return phase == llama_moe_hot_cache_graph_phase::prompt_processing;
    }
};

class llama_moe_hot_cache_pp_policy {
public:
    static llama_moe_hot_cache_pp_execution_plan build(
            llama_moe_hot_cache_graph_phase phase,
            int64_t n_tokens,
            int64_t capacity,
            bool has_cold_lane,
            int64_t n_moe_slots,
            const llama_moe_hot_cache_graph_profile & profile);

    static bool reduce_merge_enabled(int64_t n_tokens, int64_t capacity);
    static bool compact_cold_reduce_enabled(llama_moe_hot_cache_graph_phase phase, int64_t n_tokens);
    static llama_moe_hot_cache_worklist_order worklist_order(llama_moe_hot_cache_graph_phase phase);
    static bool bypass_hot_cache_for_prompt_processing(
            llm_graph_phase phase,
            bool warmup,
            int64_t n_tokens,
            int64_t default_min_tokens);
    static bool bypass_hot_cache_for_prompt_processing(
            llm_graph_phase phase,
            bool warmup,
            int64_t n_tokens,
            int64_t default_min_tokens,
            uint32_t n_hot_experts,
            uint32_t n_total_experts,
            double default_min_hot_expert_ratio);
};

const char * llama_moe_hot_cache_graph_phase_name(llama_moe_hot_cache_graph_phase phase);
