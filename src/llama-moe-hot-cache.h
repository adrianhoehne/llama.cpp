#pragma once

#include "ggml.h"
#include "ggml-cpp.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct llama_model;
struct llama_model_params;

struct llama_moe_hot_cache_entry {
    uint32_t layer = 0;
    uint32_t expert = 0;
    uint64_t hit_count = 0;
};

struct llama_moe_hot_cache_expert_size {
    uint32_t layer = 0;
    uint32_t expert = 0;
    size_t bytes = 0;
};

struct llama_moe_hot_cache_plan {
    std::vector<llama_moe_hot_cache_entry> observed;
    std::vector<llama_moe_hot_cache_expert_size> selected;
    size_t budget_bytes = 0;
    size_t used_bytes = 0;
};

struct llama_moe_hot_cache_update_stats {
    bool active = false;
    double update_rate = 0.0;
    double hit_rate = 0.0;
    uint64_t hot_slots = 0;
    uint64_t cold_slots = 0;
    size_t hot_experts = 0;
    size_t candidates = 0;
    size_t max_exchange = 0;
    size_t exchanged = 0;
    size_t layers_changed = 0;
};

struct llama_moe_hot_cache_layer {
    ggml_tensor * ffn_gate_up_exps = nullptr;
    ggml_tensor * ffn_gate_exps    = nullptr;
    ggml_tensor * ffn_up_exps      = nullptr;
    ggml_tensor * ffn_down_exps    = nullptr;
    ggml_tensor * ffn_gate_exps_s  = nullptr;
    ggml_tensor * ffn_up_exps_s    = nullptr;
    ggml_tensor * ffn_down_exps_s  = nullptr;

    ggml_tensor * hot_id_map = nullptr;
    ggml_tensor * hot_mask   = nullptr;
    ggml_tensor * cold_mask  = nullptr;

    std::vector<int32_t> hot_id_map_host;

    uint32_t n_hot = 0;
    uint32_t n_expert = 0;
    float expert_weights_scale = 0.0f;

    bool active() const {
        return n_hot > 0 && hot_id_map != nullptr && hot_mask != nullptr && cold_mask != nullptr;
    }
};

enum llama_moe_hot_cache_worklist_field : int32_t {
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID = 0,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_COUNT,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_COUNT,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COUNT,
};

struct llama_moe_hot_cache {
    std::vector<llama_moe_hot_cache_layer> layers;
    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    bool active() const {
        for (const auto & layer : layers) {
            if (layer.active()) {
                return true;
            }
        }
        return false;
    }
};

std::vector<llama_moe_hot_cache_entry> llama_moe_hot_cache_parse_perf_json(const std::string & json_str);

llama_moe_hot_cache_plan llama_moe_hot_cache_select(
        const std::vector<llama_moe_hot_cache_entry> & observed,
        const std::vector<llama_moe_hot_cache_expert_size> & sizes,
        size_t budget_bytes);

void llama_moe_hot_cache_init(llama_model & model, const llama_model_params & params);

llama_moe_hot_cache_update_stats llama_moe_hot_cache_update_from_perf_json(
        llama_model & model,
        const std::string & json_str,
        double update_rate);

bool llama_moe_hot_cache_layer_active(const llama_model & model, int il);

void llama_moe_hot_cache_build_worklist(
        ggml_tensor * dst,
        const ggml_tensor * selected_experts,
        const ggml_tensor * weights,
        const llama_moe_hot_cache_layer & layer,
        int ith,
        int nth);

void llama_moe_hot_cache_build_worklist_from_logits(
        ggml_tensor * dst,
        const ggml_tensor * logits,
        const llama_moe_hot_cache_layer & layer,
        int ith,
        int nth);
