#include "../src/moe-hot-cache/llama-moe-hot-cache-updater.h"

#include <stdexcept>
#include <string>
#include <vector>

static void require_impl(bool condition, int line) {
    if (!condition) {
        throw std::runtime_error("test assertion failed at line " + std::to_string(line));
    }
}

#define require(condition) require_impl((condition), __LINE__)

static llama_moe_hot_cache_layer make_layer(uint32_t n_expert, uint32_t n_hot, const std::vector<int32_t> & hot_id_map) {
    llama_moe_hot_cache_layer layer;
    layer.n_expert = n_expert;
    layer.n_hot = n_hot;
    layer.hot_id_map_host = hot_id_map;
    return layer;
}

static void test_current_hot_experts_returns_cache_order_and_ignores_invalid_ids() {
    const auto layer = make_layer(5, 3, { 2, -1, 0, 1, 7 });

    const auto current = llama_moe_hot_cache_current_hot_experts(layer);

    require(current.size() == 3);
    require(current[0] == 2);
    require(current[1] == 3);
    require(current[2] == 0);
}

static void test_plan_layer_replacements_pairs_best_adds_with_worst_evictions() {
    const auto layer = make_layer(5, 2, { 0, 1, -1, -1, -1 });
    const llama_moe_hot_cache_expert_hit_map hits = {
        { 0, 5   },
        { 1, 10  },
        { 2, 100 },
        { 3, 90  },
    };

    const auto candidates = llama_moe_hot_cache_plan_layer_replacements(7, layer, hits);

    require(candidates.size() == 2);
    require(candidates[0].layer == 7);
    require(candidates[0].evict_expert == 0);
    require(candidates[0].add_expert == 2);
    require(candidates[0].cache_id == 0);
    require(candidates[0].add_score == 100);
    require(candidates[0].evict_score == 5);

    require(candidates[1].layer == 7);
    require(candidates[1].evict_expert == 1);
    require(candidates[1].add_expert == 3);
    require(candidates[1].cache_id == 1);
    require(candidates[1].add_score == 90);
    require(candidates[1].evict_score == 10);
}

static void test_plan_layer_replacements_ignores_invalid_and_zero_hits() {
    const auto layer = make_layer(2, 2, { 0, 1 });
    const llama_moe_hot_cache_expert_hit_map hits = {
        { 0, 100 },
        { 1, 90  },
        { 2, 200 },
        { 3, 0   },
    };

    const auto candidates = llama_moe_hot_cache_plan_layer_replacements(0, layer, hits);

    require(candidates.empty());
}

static void test_sort_replacement_candidates_uses_gain_then_layer_then_add_expert() {
    std::vector<llama_moe_hot_cache_replacement_candidate> candidates = {
        { 2, 0, 4, 0, 100, 10 },
        { 1, 0, 5, 0, 90,  10 },
        { 1, 0, 3, 0, 100, 10 },
    };

    llama_moe_hot_cache_sort_replacement_candidates(candidates);

    require(candidates[0].layer == 1 && candidates[0].add_expert == 3);
    require(candidates[1].layer == 2 && candidates[1].add_expert == 4);
    require(candidates[2].layer == 1 && candidates[2].add_expert == 5);
}

static void test_update_max_exchange_clamps_rate_and_ceilings() {
    require(llama_moe_hot_cache_update_max_exchange(0.25, 10, 8) == 3);
    require(llama_moe_hot_cache_update_max_exchange(1.50, 10, 8) == 8);
    require(llama_moe_hot_cache_update_max_exchange(-1.0, 10, 8) == 0);
    require(llama_moe_hot_cache_update_max_exchange(0.25, 0, 8) == 0);
}

int main() {
    test_current_hot_experts_returns_cache_order_and_ignores_invalid_ids();
    test_plan_layer_replacements_pairs_best_adds_with_worst_evictions();
    test_plan_layer_replacements_ignores_invalid_and_zero_hits();
    test_sort_replacement_candidates_uses_gain_then_layer_then_add_expert();
    test_update_max_exchange_clamps_rate_and_ceilings();
    return 0;
}
