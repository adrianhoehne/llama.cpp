#pragma once

#include "llama-moe-hot-cache-adapter.h"
#include "llama-moe-hot-cache-worklist.h"

#include <cstdint>

enum class llama_moe_hot_cache_graph_phase {
    warmup,
    decode,
    prompt_processing,
};

enum class llama_moe_hot_cache_pp_cold_backend {
    cpu,
    primary,
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
    static bool dense_enabled(llama_moe_hot_cache_graph_phase phase, int64_t n_tokens);
    static bool dense_enabled(
            llama_moe_hot_cache_graph_phase phase,
            int64_t n_tokens,
            const llama_moe_hot_cache_graph_profile & profile);
    static bool weighted_cold_reduce_enabled(llama_moe_hot_cache_graph_phase phase, int64_t n_tokens);
    static llama_moe_hot_cache_pp_cold_backend cold_backend();
    static llama_moe_hot_cache_pp_cold_backend cold_backend(
            const llama_moe_hot_cache_graph_profile & profile);
    static bool hot_dummy_padding_enabled(int64_t n_tokens, bool default_enabled);
    static int64_t hot_lane_capacity(int64_t n_tokens, int64_t capacity, int64_t n_lanes, bool hot_dummy_padding);
    static llama_moe_hot_cache_worklist_order worklist_order(llama_moe_hot_cache_graph_phase phase);

    // Decides whether a model hook should enter the Hot-Cache graph for the
    // current layer. This keeps PP/TG and multi-lane gating identical across
    // all model adapters.
    static bool hot_cache_active_for_layer(
            llm_graph_phase phase,
            bool warmup,
            int64_t n_tokens,
            bool layer_active,
            bool multi_lane,
            int64_t pp_bypass_default_min_tokens = 0,
            uint32_t n_hot_experts = 0,
            uint32_t n_total_experts = 0,
            double pp_bypass_default_min_hot_expert_ratio = 0.0);

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
