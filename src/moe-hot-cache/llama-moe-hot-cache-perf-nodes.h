#pragma once

class llama_moe_layer_perf_node_classifier {
public:
    static int parse_layer_from_name(const char * name);

    static bool is_any_node(const char * name);
    static bool is_update_node(const char * name);
    static bool is_topk_node(const char * name);
    static bool is_hot_count_node(const char * name);
    static bool is_cold_count_node(const char * name);
    static bool is_hot_expert_ids_node(const char * name);
    static bool is_cold_ids_node(const char * name);

    static bool is_gate_node(const char * name);
    static bool is_up_node(const char * name);
    static bool is_down_node(const char * name);
    static bool is_hot_gate_up_node(const char * name);
    static bool is_cold_gate_up_node(const char * name);
    static bool is_hot_gate_node(const char * name);
    static bool is_cold_gate_node(const char * name);
    static bool is_hot_up_node(const char * name);
    static bool is_cold_up_node(const char * name);
    static bool is_hot_down_node(const char * name);
    static bool is_cold_down_node(const char * name);
    static bool is_hot_branch_node(const char * name);
    static bool is_cold_branch_node(const char * name);
    static bool is_worklist_node(const char * name);
    static bool is_routing_node(const char * name);
    static bool is_merge_node(const char * name);
    static bool is_hot_gather_scatter_node(const char * name);
    static bool is_cold_gather_scatter_node(const char * name);
    static bool is_hot_expert_matmul_node(const char * name);
    static bool is_cold_expert_matmul_node(const char * name);
    static bool is_expert_matmul_node(const char * name);

private:
    static bool contains(const char * name, const char * needle);
};
