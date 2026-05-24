#pragma once

#include "llama-moe-hot-cache.h"

#include <vector>

size_t llama_moe_hot_cache_tensor_expert_bytes(const ggml_tensor * t);

std::vector<llama_moe_hot_cache_expert_size> llama_moe_hot_cache_collect_expert_sizes(
        const llama_model & model);
