#include "../src/moe-hot-cache/llama-moe-hot-cache-perf-nodes.h"

#include <stdexcept>
#include <string>

static void require_impl(bool condition, int line) {
    if (!condition) {
        throw std::runtime_error("test assertion failed at line " + std::to_string(line));
    }
}

#define require(condition) require_impl((condition), __LINE__)

static void test_parse_layer_from_name() {
    require(llama_moe_layer_perf_node_classifier::parse_layer_from_name(nullptr) == -1);
    require(llama_moe_layer_perf_node_classifier::parse_layer_from_name("not_moe-1") == -1);
    require(llama_moe_layer_perf_node_classifier::parse_layer_from_name("ffn_moe_topk") == -1);
    require(llama_moe_layer_perf_node_classifier::parse_layer_from_name("ffn_moe_topk-x") == -1);
    require(llama_moe_layer_perf_node_classifier::parse_layer_from_name("ffn_moe_topk-17") == 17);
    require(llama_moe_layer_perf_node_classifier::parse_layer_from_name("blk.4.ffn_moe_hot_gate-39") == 39);
}

static void test_update_mode_nodes() {
    require(llama_moe_layer_perf_node_classifier::is_update_node("ffn_moe_topk-1"));
    require(llama_moe_layer_perf_node_classifier::is_update_node("ffn_moe_hot_count-1"));
    require(llama_moe_layer_perf_node_classifier::is_update_node("ffn_moe_cold_count-1"));
    require(llama_moe_layer_perf_node_classifier::is_update_node("ffn_moe_hot_expert_ids_compact-1"));
    require(llama_moe_layer_perf_node_classifier::is_update_node("ffn_moe_cold_ids_compact-1"));

    require(!llama_moe_layer_perf_node_classifier::is_update_node(nullptr));
    require(!llama_moe_layer_perf_node_classifier::is_update_node("ffn_moe_worklist-1"));
    require(!llama_moe_layer_perf_node_classifier::is_update_node("ffn_moe_out-1"));
}

static void test_branch_and_matmul_nodes() {
    require(llama_moe_layer_perf_node_classifier::is_hot_branch_node("ffn_moe_hot_gate_up-2"));
    require(llama_moe_layer_perf_node_classifier::is_hot_expert_matmul_node("ffn_moe_hot_gate_up-2"));
    require(!llama_moe_layer_perf_node_classifier::is_gate_node("ffn_moe_hot_gate_up-2"));
    require(!llama_moe_layer_perf_node_classifier::is_expert_matmul_node("ffn_moe_hot_gate_up-2"));

    require(llama_moe_layer_perf_node_classifier::is_gate_node("ffn_moe_hot_gate-2"));
    require(llama_moe_layer_perf_node_classifier::is_hot_expert_matmul_node("ffn_moe_hot_gate-2"));
    require(llama_moe_layer_perf_node_classifier::is_hot_branch_node("ffn_moe_hot_gate-2"));
    require(llama_moe_layer_perf_node_classifier::is_expert_matmul_node("ffn_moe_hot_gate-2"));

    require(llama_moe_layer_perf_node_classifier::is_down_node("ffn_moe_cold_down-2"));
    require(llama_moe_layer_perf_node_classifier::is_cold_expert_matmul_node("ffn_moe_cold_down-2"));
    require(llama_moe_layer_perf_node_classifier::is_cold_branch_node("ffn_moe_cold_down-2"));
    require(llama_moe_layer_perf_node_classifier::is_expert_matmul_node("ffn_moe_cold_down-2"));
}

static void test_routing_merge_and_gather_nodes() {
    require(llama_moe_layer_perf_node_classifier::is_routing_node("ffn_moe_weights_scaled-4"));
    require(llama_moe_layer_perf_node_classifier::is_routing_node("ffn_moe_topk-4"));

    require(llama_moe_layer_perf_node_classifier::is_merge_node("ffn_moe_hot_cold_slots-4"));
    require(llama_moe_layer_perf_node_classifier::is_merge_node("ffn_moe_out-4"));

    require(llama_moe_layer_perf_node_classifier::is_hot_gather_scatter_node("ffn_moe_hot_src_slots-4"));
    require(llama_moe_layer_perf_node_classifier::is_hot_gather_scatter_node("ffn_moe_hot_inputs-4"));

    require(llama_moe_layer_perf_node_classifier::is_cold_gather_scatter_node("ffn_moe_cold_token_ids-4"));
    require(llama_moe_layer_perf_node_classifier::is_cold_gather_scatter_node("ffn_moe_cold_slots-4"));

    require(!llama_moe_layer_perf_node_classifier::is_routing_node("ffn_moe_worklist-4"));
    require(!llama_moe_layer_perf_node_classifier::is_merge_node("ffn_moe_hot_src_slots-4"));
}

int main() {
    test_parse_layer_from_name();
    test_update_mode_nodes();
    test_branch_and_matmul_nodes();
    test_routing_merge_and_gather_nodes();
    return 0;
}
