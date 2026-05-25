#include "../src/moe-hot-cache/llama-moe-hot-cache-parser.h"
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

static llama_moe_hot_cache_layer make_layer(
        uint32_t n_expert,
        uint32_t n_hot,
        const std::vector<int32_t> & hot_id_map) {
    llama_moe_hot_cache_layer layer;
    layer.n_expert = n_expert;
    layer.n_hot = n_hot;
    layer.hot_id_map_host = hot_id_map;
    return layer;
}

static llama_moe_hot_cache_expert_hit_map hits_for_layer(
        const std::vector<llama_moe_hot_cache_entry> & entries,
        uint32_t layer) {
    llama_moe_hot_cache_expert_hit_map hits;
    for (const auto & entry : entries) {
        if (entry.layer == layer) {
            hits[entry.expert] += entry.hit_count;
        }
    }
    return hits;
}

static std::vector<llama_moe_hot_cache_replacement_candidate> plan_from_json(
        const std::string & json_str,
        const llama_moe_hot_cache_layer & current_layer,
        uint32_t layer) {
    const auto observations = llama_moe_hot_cache_perf_json_parser::parse_observations(json_str);
    const auto scored = llama_moe_hot_cache_weighting::score_observations(observations);
    return llama_moe_hot_cache_plan_layer_replacements(layer, current_layer, hits_for_layer(scored, layer));
}

static void test_visible_perf_json_plans_only_delta_replacements() {
    const auto current_layer = make_layer(4, 2, { 0, 1, -1, -1 });
    const std::string json_str = R"json(
{
  "enabled": true,
  "schema": "llama.cpp.moe_layer_perf.v1",
  "n_expert": 4,
  "layers": [
    {
      "layer": 0,
      "hot_experts": [[1, 100], [2, 90]],
      "cold_experts": [[0, 5], [3, 10]]
    }
  ]
}
)json";

    const auto candidates = plan_from_json(json_str, current_layer, 0);

    require(candidates.size() == 1);
    require(candidates[0].layer == 0);
    require(candidates[0].evict_expert == 0);
    require(candidates[0].add_expert == 2);
    require(candidates[0].cache_id == 0);
}

static void test_first_run_experts_json_uses_same_delta_planning() {
    const auto current_layer = make_layer(4, 2, { 0, 1, -1, -1 });
    const std::string json_str = R"json(
{
  "schema": "llama.cpp.moe_layer_opt_perf.v1",
  "n_expert": 4,
  "layers": [
    {
      "layer": 0,
      "experts": [[0, 1], [1, 20], [2, 5], [3, 40]]
    }
  ]
}
)json";

    const auto candidates = plan_from_json(json_str, current_layer, 0);

    require(candidates.size() == 1);
    require(candidates[0].evict_expert == 0);
    require(candidates[0].add_expert == 3);
    require(candidates[0].cache_id == 0);
}

static void test_runtime_apply_full_rate_can_replace_all_planned_deltas() {
    require(llama_moe_hot_cache_update_max_exchange(1.0, 2, 2) == 2);
    require(llama_moe_hot_cache_update_max_exchange(1.0, 2, 1) == 1);
}

int main() {
    test_visible_perf_json_plans_only_delta_replacements();
    test_first_run_experts_json_uses_same_delta_planning();
    test_runtime_apply_full_rate_can_replace_all_planned_deltas();
    return 0;
}
