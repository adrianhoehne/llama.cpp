#pragma once

#include "llama-moe-hot-cache.h"

#include <vector>

size_t llama_moe_hot_cache_tensor_expert_bytes(const ggml_tensor * t);

std::vector<llama_moe_hot_cache_expert_size> llama_moe_hot_cache_collect_expert_sizes(
        const llama_model & model);

llama_moe_hot_cache_device_strategy llama_moe_hot_cache_parse_device_strategy(
        const char * name);

llama_moe_hot_cache_multi_plan llama_moe_hot_cache_select_multi(
        const std::vector<llama_moe_hot_cache_entry> & observed,
        const std::vector<llama_moe_hot_cache_expert_size> & sizes,
        const std::vector<size_t> & lane_budget_bytes,
        llama_moe_hot_cache_device_strategy strategy);
