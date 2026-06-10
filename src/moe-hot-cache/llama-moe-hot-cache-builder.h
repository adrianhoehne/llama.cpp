#pragma once

#include "llama-moe-hot-cache.h"

#include "ggml-backend.h"

#include <memory>
#include <vector>

struct llama_model;

struct llama_moe_hot_cache_layer_selection_stats {
    size_t active_layers = 0;
    size_t total_hot = 0;
    size_t min_hot = 0;
    size_t max_hot = 0;

    double avg_hot() const {
        return active_layers > 0 ? double(total_hot) / double(active_layers) : 0.0;
    }
};

std::vector<std::vector<uint32_t>> llama_moe_hot_cache_group_selected_by_layer(
        const llama_moe_hot_cache_plan & plan,
        size_t n_layer);

llama_moe_hot_cache_layer_selection_stats llama_moe_hot_cache_summarize_selected_layers(
        const std::vector<std::vector<uint32_t>> & selected_by_layer);

void llama_moe_hot_cache_copy_expert_slice(
        const ggml_tensor * src,
        ggml_tensor * dst,
        uint32_t src_expert,
        uint32_t dst_expert);

void llama_moe_hot_cache_copy_scale_slice(
        const ggml_tensor * src,
        ggml_tensor * dst,
        uint32_t src_expert,
        uint32_t dst_expert);

void llama_moe_hot_cache_copy_bias_slice(
        const ggml_tensor * src,
        ggml_tensor * dst,
        uint32_t src_expert,
        uint32_t dst_expert);

void llama_moe_hot_cache_set_tensor_i32_1d(ggml_tensor * t, uint32_t index, int32_t value);
void llama_moe_hot_cache_set_tensor_f32_1d(ggml_tensor * t, uint32_t index, float value);

std::unique_ptr<llama_moe_hot_cache> llama_moe_hot_cache_build(
        const llama_model & model,
        const llama_moe_hot_cache_plan & plan,
        ggml_backend_dev_t cache_dev);

std::unique_ptr<llama_moe_hot_cache> llama_moe_hot_cache_build_multi(
        const llama_model & model,
        const llama_moe_hot_cache_multi_plan & plan,
        const std::vector<ggml_backend_dev_t> & cache_devs);
