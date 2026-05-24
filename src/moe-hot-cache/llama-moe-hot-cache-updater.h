#pragma once

#include "llama-moe-hot-cache.h"
#include "llama-moe-hot-cache-parser.h"

#include <unordered_map>
#include <vector>

struct llama_model;

struct llama_moe_hot_cache_replacement_candidate {
    uint32_t layer = 0;
    uint32_t evict_expert = 0;
    uint32_t add_expert = 0;
    uint32_t cache_id = 0;
    uint64_t add_score = 0;
    uint64_t evict_score = 0;

    uint64_t gain() const {
        return add_score > evict_score ? add_score - evict_score : 0;
    }
};

using llama_moe_hot_cache_expert_hit_map = std::unordered_map<uint32_t, uint64_t>;

std::vector<uint32_t> llama_moe_hot_cache_current_hot_experts(
        const llama_moe_hot_cache_layer & layer);

std::vector<llama_moe_hot_cache_replacement_candidate> llama_moe_hot_cache_plan_layer_replacements(
        uint32_t layer,
        const llama_moe_hot_cache_layer & cache_layer,
        const llama_moe_hot_cache_expert_hit_map & expert_hits);

void llama_moe_hot_cache_sort_replacement_candidates(
        std::vector<llama_moe_hot_cache_replacement_candidate> & candidates);

size_t llama_moe_hot_cache_update_max_exchange(
        double update_rate,
        size_t hot_experts,
        size_t candidates);

llama_moe_hot_cache_update_stats llama_moe_hot_cache_update_from_scored_observations(
        llama_model & model,
        const std::vector<llama_moe_hot_cache_perf_json_layer_slots> & layer_slots,
        const std::vector<llama_moe_hot_cache_entry> & scored_observed,
        double update_rate);
