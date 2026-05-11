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

    uint32_t n_hot = 0;
    uint32_t n_expert = 0;

    bool active() const {
        return n_hot > 0 && hot_id_map != nullptr && hot_mask != nullptr && cold_mask != nullptr;
    }
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
