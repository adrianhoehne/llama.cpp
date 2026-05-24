#pragma once

#include "../llama-arch.h"
#include "../llama-graph.h"

#include <cstdint>

struct llama_moe_hot_cache_graph_profile {
    bool cpu_decode_routing = false;
    bool decode_direct_merge = false;
    bool decode_strided_sum_rows = false;
    bool shared_input_row = false;
    bool cold_prefix_sum = false;
    bool cold_prefix_weighted_sum = false;
    bool decode_repeat_hot_input = false;
    bool cold_first_row_input = false;
    bool merge_sum_rows = false;
    bool branch_reduce_merge = false;
    int64_t cpu_decode_routing_max_tokens = 1;
    int prefix_reduce_tasks_max = 4;
};

enum class llama_moe_hot_cache_graph_kind {
    none,
    qwen35_ffn,
    logits,
};

struct llama_moe_hot_cache_model_adapter {
    llm_arch arch;
    const char * name;
    llama_moe_hot_cache_graph_kind graph_kind;
    llm_ffn_op_type ffn_op;

    llama_moe_hot_cache_graph_profile profile() const;
};

class llama_moe_hot_cache_graph_tweaks {
public:
    static int parallel_mode();
    static int64_t parallel_min_slots();
    static bool merge_sum_rows();
    static bool cpu_decode_routing();
    static bool decode_direct_merge();
    static bool decode_strided_sum_rows();
    static bool hot_dummy_padding();
    static bool shared_input_row();
    static bool cold_prefix_sum();
    static bool cold_prefix_weighted_sum();
    static bool decode_repeat_hot_input();
    static bool cold_first_row_input();
    static bool branch_reduce_merge();
    static int prefix_reduce_tasks_max();
    static bool pp_reduce_merge(int64_t n_tokens, int64_t capacity);
};

const llama_moe_hot_cache_model_adapter * llama_moe_hot_cache_find_model_adapter(llm_arch arch);
const llama_moe_hot_cache_model_adapter * llama_moe_hot_cache_find_model_adapter(
        llm_arch arch,
        llama_moe_hot_cache_graph_kind graph_kind);
bool llama_moe_hot_cache_adapter_supports_arch(llm_arch arch);
bool llama_moe_hot_cache_adapter_supports_graph_kind(
        llm_arch arch,
        llama_moe_hot_cache_graph_kind graph_kind);
const char * llama_moe_hot_cache_graph_kind_name(llama_moe_hot_cache_graph_kind graph_kind);
llama_moe_hot_cache_graph_profile llama_moe_hot_cache_graph_profile_for_arch(llm_arch arch);
