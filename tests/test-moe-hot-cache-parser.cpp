#include "../src/moe-hot-cache/llama-moe-hot-cache-parser.h"

#include <stdexcept>
#include <string>
#include <vector>

static void require_impl(bool condition, int line) {
    if (!condition) {
        throw std::runtime_error("test assertion failed at line " + std::to_string(line));
    }
}

#define require(condition) require_impl((condition), __LINE__)

static void test_parse_observations_merges_branch_counts_and_timings() {
    const std::string json = R"({
        "enabled": true,
        "schema": "llama.cpp.moe_layer_perf.v1",
        "n_expert": 4,
        "n_expert_used": 2,
        "layers": [
            {
                "layer": 2,
                "experts": [[3, 5], [1, 7], [0, 0]],
                "parallel_cold_lane_wall_time_per_call_us": 30.0,
                "parallel_hot_lane_wall_time_per_call_us": 10.0,
                "cold_slots_per_call": 4.0
            },
            {
                "layer": 1,
                "experts": [[0, 100]],
                "hot_experts": [[3, 2], [1, 10], [1, 4], [2, 0]],
                "cold_experts": [[3, 5], [2, 6]],
                "calls": 4,
                "cold_slots_total": 12,
                "parallel_join_wait_time_per_call_us": 18.0,
                "parallel_cold_lane_wall_time_per_call_us": 28.0,
                "parallel_hot_lane_wall_time_per_call_us": 8.0,
                "total_moe_time_per_call_us": 40.0
            }
        ]
    })";

    const auto observations = llama_moe_hot_cache_perf_json_parser::parse_observations(json);
    require(observations.size() == 2);

    const auto & layer1 = observations[0];
    require(layer1.layer == 1);
    require(layer1.has_branch_counts);
    require(layer1.experts.size() == 3);
    require(layer1.experts[0].expert == 1 && layer1.experts[0].hot == 14 && layer1.experts[0].cold == 0 && layer1.experts[0].raw == 0);
    require(layer1.experts[1].expert == 2 && layer1.experts[1].hot == 0 && layer1.experts[1].cold == 6 && layer1.experts[1].raw == 0);
    require(layer1.experts[2].expert == 3 && layer1.experts[2].hot == 2 && layer1.experts[2].cold == 5 && layer1.experts[2].raw == 0);
    require(layer1.cold_slots_per_call == 3.0);
    require(layer1.parallel_join_wait_time_per_call_us == 18.0);
    require(layer1.parallel_cold_lane_wall_time_per_call_us == 28.0);
    require(layer1.parallel_hot_lane_wall_time_per_call_us == 8.0);
    require(layer1.total_moe_time_per_call_us == 40.0);
    require(layer1.wait_per_cold_slot_us == 6.0);

    const auto & layer2 = observations[1];
    require(layer2.layer == 2);
    require(!layer2.has_branch_counts);
    require(layer2.experts.size() == 2);
    require(layer2.experts[0].expert == 1 && layer2.experts[0].raw == 7);
    require(layer2.experts[1].expert == 3 && layer2.experts[1].raw == 5);
    require(layer2.wait_per_cold_slot_us == 5.0);
}

static void test_parse_opt_schema_and_compat_ranking_wrapper() {
    const std::string json = R"({
        "enabled": true,
        "schema": "llama.cpp.moe_layer_opt_perf.v1",
        "n_expert": 4,
        "n_expert_used": 16,
        "summary": {
            "layer_calls": 8,
            "hot_slot_ratio": 0
        },
        "layers": [
            {
                "layer": 0,
                "calls": 4,
                "experts": [[0, 2], [1, 5], [2, 0], [3, 4]],
                "hot_experts": [],
                "cold_experts": []
            },
            {
                "layer": 1,
                "calls": 4,
                "experts": [[0, 1], [1, 6], [2, 3], [3, 0]],
                "hot_experts": [],
                "cold_experts": []
            }
        ]
    })";

    const auto observations = llama_moe_hot_cache_perf_json_parser::parse_observations(json);
    require(observations.size() == 2);
    require(observations[0].layer == 0);
    require(observations[1].layer == 1);

    const auto entries = llama_moe_hot_cache_parse_perf_json(json);
    require(entries.size() == 6);
    require(entries[0].layer == 1 && entries[0].expert == 1 && entries[0].hit_count == 6);
    require(entries[1].layer == 0 && entries[1].expert == 1 && entries[1].hit_count == 5);
    require(entries[2].layer == 0 && entries[2].expert == 3 && entries[2].hit_count == 4);
    require(entries[3].layer == 1 && entries[3].expert == 2 && entries[3].hit_count == 3);
    require(entries[4].layer == 0 && entries[4].expert == 0 && entries[4].hit_count == 2);
    require(entries[5].layer == 1 && entries[5].expert == 0 && entries[5].hit_count == 1);
}

static void test_parse_enabled_layer_slots() {
    const std::string json = R"({
        "enabled": true,
        "schema": "llama.cpp.moe_layer_perf.v1",
        "layers": [
            { "layer": 1, "hot_slots_total": 10, "cold_slots_total": 2 },
            { "layer": "ignored", "hot_slots_total": 99, "cold_slots_total": 99 },
            { "layer": 3, "hot_slots_total": 7 }
        ]
    })";

    std::vector<llama_moe_hot_cache_perf_json_layer_slots> slots;
    require(llama_moe_hot_cache_perf_json_parser::parse_enabled_layer_slots(json, slots));
    require(slots.size() == 2);
    require(slots[0].layer == 1 && slots[0].hot_slots == 10 && slots[0].cold_slots == 2);
    require(slots[1].layer == 3 && slots[1].hot_slots == 7 && slots[1].cold_slots == 0);

    const std::string disabled = R"({
        "enabled": false,
        "schema": "llama.cpp.moe_layer_perf.v1",
        "layers": [{ "layer": 0, "hot_slots_total": 1, "cold_slots_total": 1 }]
    })";
    require(!llama_moe_hot_cache_perf_json_parser::parse_enabled_layer_slots(disabled, slots));
    require(slots.empty());
}

static void test_parser_rejects_invalid_json() {
    bool threw = false;
    try {
        (void) llama_moe_hot_cache_perf_json_parser::parse_observations(R"({"schema":"wrong","layers":[]})");
    } catch (const std::runtime_error &) {
        threw = true;
    }
    require(threw);

    threw = false;
    try {
        (void) llama_moe_hot_cache_perf_json_parser::parse_observations(R"({
            "enabled": true,
            "schema": "llama.cpp.moe_layer_perf.v1",
            "layers": [{ "layer": 0, "experts": [[1]] }]
        })");
    } catch (const std::runtime_error &) {
        threw = true;
    }
    require(threw);
}

int main() {
    test_parse_observations_merges_branch_counts_and_timings();
    test_parse_opt_schema_and_compat_ranking_wrapper();
    test_parse_enabled_layer_slots();
    test_parser_rejects_invalid_json();
    return 0;
}
