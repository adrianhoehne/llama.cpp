#include "llama-moe-hot-cache-perf-nodes.h"

#include <cstdlib>
#include <cstring>

int llama_moe_layer_perf_node_classifier::parse_layer_from_name(const char * name) {
    if (name == nullptr) {
        return -1;
    }

    const char * p = std::strstr(name, "ffn_moe_");
    if (p == nullptr) {
        return -1;
    }

    const char * dash = std::strchr(p, '-');
    if (dash == nullptr) {
        return -1;
    }

    dash++;

    if (*dash < '0' || *dash > '9') {
        return -1;
    }

    return std::atoi(dash);
}

bool llama_moe_layer_perf_node_classifier::contains(const char * name, const char * needle) {
    return name != nullptr && std::strstr(name, needle) != nullptr;
}

bool llama_moe_layer_perf_node_classifier::is_any_node(const char * name) {
    return contains(name, "ffn_moe_");
}

bool llama_moe_layer_perf_node_classifier::is_topk_node(const char * name) {
    return contains(name, "ffn_moe_topk-");
}

bool llama_moe_layer_perf_node_classifier::is_hot_count_node(const char * name) {
    return contains(name, "ffn_moe_hot_count-");
}

bool llama_moe_layer_perf_node_classifier::is_cold_count_node(const char * name) {
    return contains(name, "ffn_moe_cold_count-");
}

bool llama_moe_layer_perf_node_classifier::is_hot_expert_ids_node(const char * name) {
    return contains(name, "ffn_moe_hot_expert_ids_compact-");
}

bool llama_moe_layer_perf_node_classifier::is_cold_ids_node(const char * name) {
    return contains(name, "ffn_moe_cold_ids_compact-");
}

bool llama_moe_layer_perf_node_classifier::is_update_node(const char * name) {
    return is_topk_node(name) ||
           is_hot_count_node(name) ||
           is_cold_count_node(name) ||
           is_hot_expert_ids_node(name) ||
           is_cold_ids_node(name);
}

bool llama_moe_layer_perf_node_classifier::is_gate_node(const char * name) {
    return contains(name, "ffn_moe_gate-") ||
           contains(name, "ffn_moe_hot_gate-") ||
           contains(name, "ffn_moe_cold_gate-");
}

bool llama_moe_layer_perf_node_classifier::is_up_node(const char * name) {
    return contains(name, "ffn_moe_up-") ||
           contains(name, "ffn_moe_hot_up-") ||
           contains(name, "ffn_moe_cold_up-");
}

bool llama_moe_layer_perf_node_classifier::is_down_node(const char * name) {
    return contains(name, "ffn_moe_down-") ||
           contains(name, "ffn_moe_hot_down-") ||
           contains(name, "ffn_moe_cold_down-");
}

bool llama_moe_layer_perf_node_classifier::is_hot_gate_up_node(const char * name) {
    return contains(name, "ffn_moe_hot_gate_up-");
}

bool llama_moe_layer_perf_node_classifier::is_cold_gate_up_node(const char * name) {
    return contains(name, "ffn_moe_cold_gate_up-");
}

bool llama_moe_layer_perf_node_classifier::is_hot_gate_node(const char * name) {
    return contains(name, "ffn_moe_hot_gate-");
}

bool llama_moe_layer_perf_node_classifier::is_cold_gate_node(const char * name) {
    return contains(name, "ffn_moe_cold_gate-");
}

bool llama_moe_layer_perf_node_classifier::is_hot_up_node(const char * name) {
    return contains(name, "ffn_moe_hot_up-");
}

bool llama_moe_layer_perf_node_classifier::is_cold_up_node(const char * name) {
    return contains(name, "ffn_moe_cold_up-");
}

bool llama_moe_layer_perf_node_classifier::is_hot_down_node(const char * name) {
    return contains(name, "ffn_moe_hot_down-");
}

bool llama_moe_layer_perf_node_classifier::is_cold_down_node(const char * name) {
    return contains(name, "ffn_moe_cold_down-");
}

bool llama_moe_layer_perf_node_classifier::is_hot_branch_node(const char * name) {
    return contains(name, "ffn_moe_hot_");
}

bool llama_moe_layer_perf_node_classifier::is_cold_branch_node(const char * name) {
    return contains(name, "ffn_moe_cold_");
}

bool llama_moe_layer_perf_node_classifier::is_worklist_node(const char * name) {
    return contains(name, "ffn_moe_worklist-");
}

bool llama_moe_layer_perf_node_classifier::is_routing_node(const char * name) {
    return contains(name, "ffn_moe_logits-") ||
           contains(name, "ffn_moe_probs-") ||
           contains(name, "ffn_moe_argsort-") ||
           contains(name, "ffn_moe_topk-") ||
           contains(name, "ffn_moe_weights-") ||
           contains(name, "ffn_moe_weights_sum-") ||
           contains(name, "ffn_moe_weights_sum_clamped-") ||
           contains(name, "ffn_moe_weights_norm-") ||
           contains(name, "ffn_moe_weights_scaled-");
}

bool llama_moe_layer_perf_node_classifier::is_merge_node(const char * name) {
    return contains(name, "ffn_moe_hot_cold_slots-") ||
           contains(name, "ffn_moe_out-");
}

bool llama_moe_layer_perf_node_classifier::is_hot_gather_scatter_node(const char * name) {
    return contains(name, "ffn_moe_hot_ids_compact-") ||
           contains(name, "ffn_moe_hot_expert_ids_compact-") ||
           contains(name, "ffn_moe_hot_src_slots-") ||
           contains(name, "ffn_moe_hot_token_ids-") ||
           contains(name, "ffn_moe_hot_weights_compact-") ||
           contains(name, "ffn_moe_hot_inputs-") ||
           contains(name, "ffn_moe_hot_slots-");
}

bool llama_moe_layer_perf_node_classifier::is_cold_gather_scatter_node(const char * name) {
    return contains(name, "ffn_moe_cold_ids_compact-") ||
           contains(name, "ffn_moe_cold_src_slots-") ||
           contains(name, "ffn_moe_cold_token_ids-") ||
           contains(name, "ffn_moe_cold_weights_compact-") ||
           contains(name, "ffn_moe_cold_inputs-") ||
           contains(name, "ffn_moe_cold_slots-");
}

bool llama_moe_layer_perf_node_classifier::is_hot_expert_matmul_node(const char * name) {
    return is_hot_gate_up_node(name) ||
           is_hot_gate_node(name)    ||
           is_hot_up_node(name)      ||
           is_hot_down_node(name);
}

bool llama_moe_layer_perf_node_classifier::is_cold_expert_matmul_node(const char * name) {
    return is_cold_gate_up_node(name) ||
           is_cold_gate_node(name)    ||
           is_cold_up_node(name)      ||
           is_cold_down_node(name);
}

bool llama_moe_layer_perf_node_classifier::is_expert_matmul_node(const char * name) {
    return is_gate_node(name) ||
           is_up_node(name)   ||
           is_down_node(name);
}
