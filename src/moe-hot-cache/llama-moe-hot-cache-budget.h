#pragma once

#include "llama-moe-hot-cache.h"

#include "ggml-backend.h"

ggml_backend_dev_t llama_moe_hot_cache_select_gpu_dev(const llama_model * model = nullptr);

ggml_backend_dev_t llama_moe_hot_cache_resolve_gpu_dev(
        const llama_model * model,
        const char * name);

size_t llama_moe_hot_cache_compute_auto_budget_bytes(
        size_t free_bytes,
        size_t kv_reserve_bytes,
        size_t safety_reserve_mib);

size_t llama_moe_hot_cache_auto_budget_bytes(
        const llama_model & model,
        const llama_model_params & params,
        ggml_backend_dev_t dev,
        bool reserve_kv_cache);

size_t llama_moe_hot_cache_auto_budget_bytes(
        const llama_model & model,
        const llama_model_params & params,
        ggml_backend_dev_t dev,
        bool reserve_kv_cache,
        uint64_t reserve_mib);
