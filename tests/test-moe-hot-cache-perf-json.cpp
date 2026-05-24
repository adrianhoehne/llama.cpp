#include "../src/moe-hot-cache/llama-moe-hot-cache-perf-json.h"

#include <stdexcept>
#include <string>

static void require_impl(bool condition, int line) {
    if (!condition) {
        throw std::runtime_error("test assertion failed at line " + std::to_string(line));
    }
}

#define require(condition) require_impl((condition), __LINE__)

static bool contains(const std::string & text, const char * needle) {
    return text.find(needle) != std::string::npos;
}

static void test_disabled_json_is_minimal() {
    const std::string json = llama_moe_layer_perf_json_serializer::serialize_disabled();
    require(json == "{\"enabled\":false,\"mode\":\"off\",\"schema\":\"llama.cpp.moe_layer_opt_perf.v1\",\"layers\":[]}");
}

static void test_update_mode_keeps_only_update_metrics() {
    llama_moe_layer_perf_json_snapshot snapshot;
    snapshot.mode = LLAMA_MOE_LAYER_PERF_MODE_UPDATE;
    snapshot.n_expert = 4;
    snapshot.n_expert_used = 2;
    snapshot.updates = 3;

    llama_moe_layer_perf_json_layer_snapshot layer;
    layer.calls = 2;
    layer.hot_slots_total = 6;
    layer.cold_slots_total = 4;
    layer.hot_worklist_calls = 2;
    layer.cold_worklist_calls = 2;
    layer.parallel_hot_lane_wall_time_us = 20;
    layer.parallel_cold_lane_wall_time_us = 40;
    layer.parallel_join_wait_time_us = 10;
    layer.total_moe_time_us = 100;
    layer.experts = { 0, 5, 0, 2 };
    layer.hot_experts = { 0, 3, 0, 0 };
    layer.cold_experts = { 0, 0, 0, 2 };
    snapshot.layers.push_back(layer);

    const std::string json = llama_moe_layer_perf_json_serializer::serialize(snapshot);
    require(contains(json, "\"enabled\":true"));
    require(contains(json, "\"mode\":\"update\""));
    require(contains(json, "\"n_expert\":4"));
    require(contains(json, "\"updates\":3"));
    require(contains(json, "\"hot_slot_ratio\":0.6"));
    require(contains(json, "\"parallel_hot_lane_wall_time_per_call_us\":10"));
    require(contains(json, "\"parallel_cold_lane_wall_time_per_call_us\":20"));
    require(contains(json, "\"parallel_join_wait_time_per_call_us\":5"));
    require(contains(json, "\"experts\":[[1,5],[3,2]]"));
    require(contains(json, "\"hot_experts\":[[1,3]]"));
    require(contains(json, "\"cold_experts\":[[3,2]]"));
    require(!contains(json, "\"total_moe_time_per_call_us\""));
    require(!contains(json, "\"parallel_fallbacks\""));
}

static void test_full_mode_serializes_raw_counts_as_cold() {
    llama_moe_layer_perf_json_snapshot snapshot;
    snapshot.mode = LLAMA_MOE_LAYER_PERF_MODE_FULL;
    snapshot.n_expert = 4;
    snapshot.n_expert_used = 2;

    llama_moe_layer_perf_json_layer_snapshot layer;
    layer.calls = 2;
    layer.expert_hits_total = 8;
    layer.total_moe_time_us = 120;
    layer.routing_time_us = 12;
    layer.experts = { 0, 5, 0, 3 };
    layer.hot_experts = { 0, 0, 0, 0 };
    layer.cold_experts = { 0, 0, 0, 0 };
    snapshot.layers.push_back(layer);

    const std::string json = llama_moe_layer_perf_json_serializer::serialize(snapshot);
    require(contains(json, "\"mode\":\"full\""));
    require(contains(json, "\"hot_slots_total\":0"));
    require(contains(json, "\"cold_slots_total\":8"));
    require(contains(json, "\"hot_slots_per_call\":0"));
    require(contains(json, "\"cold_slots_per_call\":4"));
    require(contains(json, "\"total_moe_time_per_call_us\":60"));
    require(contains(json, "\"routing_time_per_call_us\":6"));
    require(contains(json, "\"cold_experts\":[[1,5],[3,3]]"));
}

static void test_full_mode_serializes_parallel_debug() {
    llama_moe_layer_perf_json_snapshot snapshot;
    snapshot.mode = LLAMA_MOE_LAYER_PERF_MODE_FULL;

    llama_moe_layer_perf_json_layer_snapshot layer;
    layer.calls = 1;
    layer.hot_slots_total = 1;
    layer.cold_slots_total = 1;
    layer.parallel_fallbacks = 2;
    layer.parallel_fallback_bad_split_order = 1;
    layer.parallel_fallback_threshold = 1;
    layer.parallel_split_debug_samples = 1;
    layer.parallel_split_debug_hot_begin = 2;
    layer.parallel_split_debug_hot_end = 5;
    layer.parallel_split_debug_cold_begin = 5;
    layer.parallel_split_debug_cold_end = 8;
    layer.parallel_split_debug_join = 9;
    layer.parallel_split_debug_hot_count = 3;
    layer.parallel_split_debug_cold_count = 4;
    layer.parallel_split_debug_hot_backend = 0;
    layer.parallel_split_debug_cold_backend = 1;
    layer.parallel_split_debug_join_backend = 0;
    snapshot.layers.push_back(layer);

    const std::string json = llama_moe_layer_perf_json_serializer::serialize(snapshot);
    require(contains(json, "\"parallel_fallbacks\":2"));
    require(contains(json, "\"parallel_fallback_reasons\":{\"bad_split_order\":1,\"threshold\":1}"));
    require(contains(json, "\"parallel_split_debug\":{\"samples\":1"));
    require(contains(json, "\"hot\":[2,5]"));
    require(contains(json, "\"cold\":[5,8]"));
    require(contains(json, "\"counts\":{\"hot\":3,\"cold\":4}"));
    require(contains(json, "\"backends\":{\"hot\":0,\"cold\":1,\"join\":0}"));
}

int main() {
    test_disabled_json_is_minimal();
    test_update_mode_keeps_only_update_metrics();
    test_full_mode_serializes_raw_counts_as_cold();
    test_full_mode_serializes_parallel_debug();
    return 0;
}
