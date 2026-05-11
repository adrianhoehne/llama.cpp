#pragma once

#include "ggml.h"

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

std::vector<llama_moe_hot_cache_entry> llama_moe_hot_cache_parse_perf_json(const std::string & json_str);

llama_moe_hot_cache_plan llama_moe_hot_cache_select(
        const std::vector<llama_moe_hot_cache_entry> & observed,
        const std::vector<llama_moe_hot_cache_expert_size> & sizes,
        size_t budget_bytes);

void llama_moe_hot_cache_init(const llama_model & model, const llama_model_params & params);
