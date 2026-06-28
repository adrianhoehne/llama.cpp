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

int llama_moe_layer_perf_node_classifier::hot_lane_from_name(const char * name) {
    const char * p = name == nullptr ? nullptr : std::strstr(name, "ffn_moe_hot");
    if (p == nullptr) {
        return -1;
    }

    p += std::strlen("ffn_moe_hot");
    if (*p < '0' || *p > '9') {
        return -1;
    }

    const int lane = *p - '0';
    ++p;
    if (lane < 0 || lane >= 3) {
        return -1;
    }

    return *p == '_' || *p == '-' || *p == '\0' ? lane : -1;
}

bool llama_moe_layer_perf_node_classifier::contains(const char * name, const char * needle) {
    return name != nullptr && std::strstr(name, needle) != nullptr;
}

bool llama_moe_layer_perf_node_classifier::has_node_base(const char * name, const char * base) {
    const char * p = name;
    const size_t base_len = std::strlen(base);

    while (p != nullptr && (p = std::strstr(p, base)) != nullptr) {
        const char next = p[base_len];
        if (next == '\0' || next == '-' || next == '_') {
            return true;
        }
        p += base_len;
    }

    return false;
}

const char * llama_moe_layer_perf_node_classifier::hot_branch_suffix(const char * name) {
    const char * p = name == nullptr ? nullptr : std::strstr(name, "ffn_moe_hot");
    if (p == nullptr) {
        return nullptr;
    }

    p += std::strlen("ffn_moe_hot");
    while (*p >= '0' && *p <= '9') {
        ++p;
    }

    if (*p != '_') {
        return nullptr;
    }

    ++p;
    if (std::strncmp(p, "cold_", 5) == 0) {
        return nullptr;
    }

    return p;
}

const char * llama_moe_layer_perf_node_classifier::cold_branch_suffix(const char * name) {
    const char * p = name == nullptr ? nullptr : std::strstr(name, "ffn_moe_cold_");
    if (p == nullptr) {
        return nullptr;
    }

    return p + std::strlen("ffn_moe_cold_");
}

bool llama_moe_layer_perf_node_classifier::branch_has_component_base(const char * branch_suffix, const char * component) {
    const size_t component_len = std::strlen(component);
    const char * p = branch_suffix;

    while (p != nullptr && (p = std::strstr(p, component)) != nullptr) {
        const bool starts_at_component = p == branch_suffix || p[-1] == '_';
        const bool ends_at_component = p[component_len] == '\0' || p[component_len] == '-';
        const bool is_gate_up_when_matching_up =
            std::strcmp(component, "up") == 0 &&
            p >= branch_suffix + 5 &&
            std::strncmp(p - 5, "gate_", 5) == 0;

        if (starts_at_component && ends_at_component && !is_gate_up_when_matching_up) {
            return true;
        }

        p += component_len;
    }

    return false;
}

bool llama_moe_layer_perf_node_classifier::branch_has_component_prefix(const char * branch_suffix, const char * component) {
    const size_t component_len = std::strlen(component);
    const char * p = branch_suffix;

    while (p != nullptr && (p = std::strstr(p, component)) != nullptr) {
        const bool starts_at_component = p == branch_suffix || p[-1] == '_';
        const char next = p[component_len];
        const bool has_component_boundary = next == '\0' || next == '-' || next == '_';
        const bool is_qualified_ids =
            std::strcmp(component, "ids") == 0 &&
            ((p >= branch_suffix + 4 && std::strncmp(p - 4, "src_", 4) == 0) ||
             (p >= branch_suffix + 6 && std::strncmp(p - 6, "token_", 6) == 0) ||
             (p >= branch_suffix + 6 && std::strncmp(p - 6, "scale_", 6) == 0) ||
             (p >= branch_suffix + 7 && std::strncmp(p - 7, "expert_", 7) == 0));

        if (starts_at_component && has_component_boundary && !is_qualified_ids) {
            return true;
        }

        p += component_len;
    }

    return false;
}

bool llama_moe_layer_perf_node_classifier::branch_has_activation(const char * branch_suffix) {
    return branch_has_component_prefix(branch_suffix, "silu")   ||
           branch_has_component_prefix(branch_suffix, "swiglu") ||
           branch_has_component_prefix(branch_suffix, "gelu")   ||
           branch_has_component_prefix(branch_suffix, "geglu")  ||
           branch_has_component_prefix(branch_suffix, "relu")   ||
           branch_has_component_prefix(branch_suffix, "reglu");
}

bool llama_moe_layer_perf_node_classifier::is_any_node(const char * name) {
    return contains(name, "ffn_moe_");
}

bool llama_moe_layer_perf_node_classifier::is_topk_node(const char * name) {
    return has_node_base(name, "ffn_moe_topk");
}

bool llama_moe_layer_perf_node_classifier::is_hot_count_node(const char * name) {
    return has_node_base(name, "ffn_moe_hot_count");
}

bool llama_moe_layer_perf_node_classifier::is_hot_lane_count_node(const char * name) {
    const char * p = name == nullptr ? nullptr : std::strstr(name, "ffn_moe_hot");
    if (p == nullptr) {
        return false;
    }

    p += std::strlen("ffn_moe_hot");
    if (*p < '0' || *p > '2') {
        return false;
    }

    ++p;
    if (std::strncmp(p, "_count", 6) != 0) {
        return false;
    }

    const char next = p[6];
    return next == '\0' || next == '-' || next == '_';
}

bool llama_moe_layer_perf_node_classifier::is_cold_count_node(const char * name) {
    return has_node_base(name, "ffn_moe_cold_count");
}

bool llama_moe_layer_perf_node_classifier::is_hot_expert_ids_node(const char * name) {
    const char * suffix = hot_branch_suffix(name);
    return suffix != nullptr && branch_has_component_prefix(suffix, "expert_ids");
}

bool llama_moe_layer_perf_node_classifier::is_cold_ids_node(const char * name) {
    const char * suffix = cold_branch_suffix(name);
    return suffix != nullptr && branch_has_component_prefix(suffix, "ids");
}

bool llama_moe_layer_perf_node_classifier::is_multi_pp_worklist_node(const char * name) {
    return is_worklist_node(name) &&
           contains(name, "multi_pp") &&
           !contains(name, " (");
}

bool llama_moe_layer_perf_node_classifier::is_update_node(const char * name) {
    return is_topk_node(name) ||
           is_hot_count_node(name) ||
           is_hot_lane_count_node(name) ||
           is_cold_count_node(name) ||
           is_hot_expert_ids_node(name) ||
           is_cold_ids_node(name) ||
           is_multi_pp_worklist_node(name);
}

bool llama_moe_layer_perf_node_classifier::is_gate_node(const char * name) {
    return has_node_base(name, "ffn_moe_gate") ||
           branch_has_component_base(hot_branch_suffix(name), "gate") ||
           branch_has_component_base(cold_branch_suffix(name), "gate");
}

bool llama_moe_layer_perf_node_classifier::is_gate_up_node(const char * name) {
    return has_node_base(name, "ffn_moe_gate_up") ||
           is_hot_gate_up_node(name) ||
           is_cold_gate_up_node(name);
}

bool llama_moe_layer_perf_node_classifier::is_up_node(const char * name) {
    return has_node_base(name, "ffn_moe_up") ||
           branch_has_component_base(hot_branch_suffix(name), "up") ||
           branch_has_component_base(cold_branch_suffix(name), "up");
}

bool llama_moe_layer_perf_node_classifier::is_down_node(const char * name) {
    return has_node_base(name, "ffn_moe_down") ||
           branch_has_component_base(hot_branch_suffix(name), "down") ||
           branch_has_component_base(cold_branch_suffix(name), "down");
}

bool llama_moe_layer_perf_node_classifier::is_activation_node(const char * name) {
    return has_node_base(name, "ffn_moe_silu")       ||
           has_node_base(name, "ffn_moe_swiglu")     ||
           has_node_base(name, "ffn_moe_gelu")       ||
           has_node_base(name, "ffn_moe_geglu")      ||
           has_node_base(name, "ffn_moe_relu")       ||
           has_node_base(name, "ffn_moe_reglu")      ||
           branch_has_activation(hot_branch_suffix(name)) ||
           branch_has_activation(cold_branch_suffix(name));
}

bool llama_moe_layer_perf_node_classifier::is_hot_gate_up_node(const char * name) {
    return branch_has_component_base(hot_branch_suffix(name), "gate_up");
}

bool llama_moe_layer_perf_node_classifier::is_cold_gate_up_node(const char * name) {
    return branch_has_component_base(cold_branch_suffix(name), "gate_up");
}

bool llama_moe_layer_perf_node_classifier::is_hot_gate_node(const char * name) {
    return branch_has_component_base(hot_branch_suffix(name), "gate");
}

bool llama_moe_layer_perf_node_classifier::is_cold_gate_node(const char * name) {
    return branch_has_component_base(cold_branch_suffix(name), "gate");
}

bool llama_moe_layer_perf_node_classifier::is_hot_up_node(const char * name) {
    return branch_has_component_base(hot_branch_suffix(name), "up");
}

bool llama_moe_layer_perf_node_classifier::is_cold_up_node(const char * name) {
    return branch_has_component_base(cold_branch_suffix(name), "up");
}

bool llama_moe_layer_perf_node_classifier::is_hot_down_node(const char * name) {
    return branch_has_component_base(hot_branch_suffix(name), "down");
}

bool llama_moe_layer_perf_node_classifier::is_cold_down_node(const char * name) {
    return branch_has_component_base(cold_branch_suffix(name), "down");
}

bool llama_moe_layer_perf_node_classifier::is_hot_activation_node(const char * name) {
    return branch_has_activation(hot_branch_suffix(name));
}

bool llama_moe_layer_perf_node_classifier::is_cold_activation_node(const char * name) {
    return branch_has_activation(cold_branch_suffix(name));
}

bool llama_moe_layer_perf_node_classifier::is_hot_branch_node(const char * name) {
    return hot_branch_suffix(name) != nullptr;
}

bool llama_moe_layer_perf_node_classifier::is_cold_branch_node(const char * name) {
    return cold_branch_suffix(name) != nullptr;
}

bool llama_moe_layer_perf_node_classifier::is_worklist_node(const char * name) {
    return has_node_base(name, "ffn_moe_worklist");
}

bool llama_moe_layer_perf_node_classifier::is_routing_node(const char * name) {
    return has_node_base(name, "ffn_moe_logits") ||
           has_node_base(name, "ffn_moe_probs") ||
           has_node_base(name, "ffn_moe_argsort") ||
           has_node_base(name, "ffn_moe_topk") ||
           has_node_base(name, "ffn_moe_weights") ||
           has_node_base(name, "ffn_moe_weights_sum") ||
           has_node_base(name, "ffn_moe_weights_sum_clamped") ||
           has_node_base(name, "ffn_moe_weights_norm") ||
           has_node_base(name, "ffn_moe_weights_scaled");
}

bool llama_moe_layer_perf_node_classifier::is_merge_node(const char * name) {
    return has_node_base(name, "ffn_moe_hot_cold_slots") ||
           has_node_base(name, "ffn_moe_out");
}

bool llama_moe_layer_perf_node_classifier::is_join_node(const char * name) {
    return has_node_base(name, "ffn_moe_join_hot") ||
           has_node_base(name, "ffn_moe_join_cold");
}

bool llama_moe_layer_perf_node_classifier::is_hot_join_node(const char * name) {
    return has_node_base(name, "ffn_moe_join_hot");
}

bool llama_moe_layer_perf_node_classifier::is_cold_join_node(const char * name) {
    return has_node_base(name, "ffn_moe_join_cold");
}

bool llama_moe_layer_perf_node_classifier::is_hot_gather_scatter_node(const char * name) {
    if (contains(name, "slots_reduced")) {
        return false;
    }

    const char * suffix = hot_branch_suffix(name);
    return branch_has_component_prefix(suffix, "ids") ||
           branch_has_component_prefix(suffix, "expert_ids") ||
           branch_has_component_prefix(suffix, "src_slots") ||
           branch_has_component_prefix(suffix, "token_ids") ||
           branch_has_component_prefix(suffix, "weights") ||
           branch_has_component_prefix(suffix, "inputs") ||
           branch_has_component_prefix(suffix, "slots");
}

bool llama_moe_layer_perf_node_classifier::is_cold_gather_scatter_node(const char * name) {
    if (contains(name, "slots_reduced")) {
        return false;
    }

    const char * suffix = cold_branch_suffix(name);
    return branch_has_component_prefix(suffix, "ids") ||
           branch_has_component_prefix(suffix, "src_slots") ||
           branch_has_component_prefix(suffix, "token_ids") ||
           branch_has_component_prefix(suffix, "weights") ||
           branch_has_component_prefix(suffix, "inputs") ||
           branch_has_component_prefix(suffix, "slots");
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
    return is_gate_up_node(name) ||
           is_gate_node(name)    ||
           is_up_node(name)   ||
           is_down_node(name);
}
