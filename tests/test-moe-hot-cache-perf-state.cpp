#include "../src/moe-hot-cache/llama-moe-hot-cache-perf-state.h"

#include <limits>
#include <stdexcept>
#include <string>

static void require_impl(bool condition, int line) {
    if (!condition) {
        throw std::runtime_error("test assertion failed at line " + std::to_string(line));
    }
}

#define require(condition) require_impl((condition), __LINE__)

static void test_ensure_shape_preserves_existing_counts() {
    llama_moe_layer_perf_state state;
    state.ensure_shape_locked(1, 2, 1);
    state.layers[0].experts[1] = 7;
    state.layers[0].hot_experts[0] = 3;

    state.ensure_shape_locked(2, 4, 2);

    require(state.n_expert == 4);
    require(state.n_expert_used == 2);
    require(state.layers.size() == 2);
    require(state.layers[0].experts.size() == 4);
    require(state.layers[0].hot_experts.size() == 4);
    require(state.layers[0].cold_experts.size() == 4);
    require(state.layers[0].experts[1] == 7);
    require(state.layers[0].hot_experts[0] == 3);
    require(state.layers[0].experts[3] == 0);
}

static void test_begin_and_end_window() {
    llama_moe_layer_perf_state state;
    state.active = true;
    state.begin_locked(0, 4, 2);
    require(!state.active);
    require(state.updates == 0);

    state.begin_locked(2, 4, 2);
    require(state.active);
    require(state.updates == 1);
    require(state.last_callback_us == 0);
    require(state.layers.size() == 2);

    state.last_callback_us = 42;
    state.end_locked();
    require(!state.active);
    require(state.last_callback_us == 0);
}

static void test_add_expert_counts_ignore_invalid_ids() {
    llama_moe_layer_perf_state state;
    state.ensure_shape_locked(1, 3, 2);

    state.add_expert_locked(0, 1);
    state.add_expert_locked(0, 1);
    state.add_expert_locked(0, 3);
    state.add_expert_locked(1, 1);

    require(state.layers[0].experts[1] == 2);
    require(state.layers[0].expert_hits_total == 2);

    state.add_branch_expert_locked(0, 2, true);
    state.add_branch_expert_locked(0, 2, false);
    state.add_branch_expert_locked(0, 4, true);
    state.add_branch_expert_locked(3, 2, false);

    require(state.layers[0].hot_experts[2] == 1);
    require(state.layers[0].cold_experts[2] == 1);
}

static void test_reset_clears_counters_and_preserves_shape() {
    llama_moe_layer_perf_state state;
    state.ensure_shape_locked(1, 3, 2);
    auto & layer = state.layers[0];
    layer.calls = 5;
    layer.expert_hits_total = 6;
    layer.hot_slots_total = 7;
    layer.cold_slots_total = 8;
    layer.hot_lane_slots_total[1] = 28;
    layer.hot_lane_worklist_calls[1] = 29;
    layer.hot_lane_zero_calls[1] = 30;
    layer.total_moe_time_us = 9;
    layer.hot_lane_branch_time_us[1] = 31;
    layer.hot_lane_expert_matmul_time_us[1] = 32;
    layer.hot_lane_gather_scatter_time_us[1] = 33;
    layer.join_time_us = 34;
    layer.hot_join_time_us = 35;
    layer.cold_join_time_us = 36;
    layer.parallel_fallbacks = 10;
    layer.parallel_split_debug_samples = 11;
    layer.parallel_split_debug_hot_begin = 12;
    layer.parallel_split_debug_hot_end = 13;
    layer.parallel_split_debug_cold_begin = 14;
    layer.parallel_split_debug_cold_end = 15;
    layer.parallel_split_debug_join = 16;
    layer.parallel_split_debug_hot_count = 17;
    layer.parallel_split_debug_cold_count = 18;
    layer.parallel_split_debug_hot_backend = 19;
    layer.parallel_split_debug_cold_backend = 20;
    layer.parallel_split_debug_join_backend = 21;
    layer.gate_time_us = 22;
    layer.up_time_us = 23;
    layer.down_time_us = 24;
    layer.experts[1] = 25;
    layer.hot_experts[2] = 26;
    layer.cold_experts[0] = 27;
    state.updates = 3;
    state.last_callback_us = 99;

    require(state.has_data_locked());
    state.reset_locked(true);

    require(state.layers.size() == 1);
    require(state.layers[0].experts.size() == 3);
    require(state.layers[0].calls == 0);
    require(state.layers[0].expert_hits_total == 0);
    require(state.layers[0].total_moe_time_us == 0);
    require(state.layers[0].hot_lane_slots_total[1] == 0);
    require(state.layers[0].hot_lane_worklist_calls[1] == 0);
    require(state.layers[0].hot_lane_zero_calls[1] == 0);
    require(state.layers[0].hot_lane_branch_time_us[1] == 0);
    require(state.layers[0].hot_lane_expert_matmul_time_us[1] == 0);
    require(state.layers[0].hot_lane_gather_scatter_time_us[1] == 0);
    require(state.layers[0].join_time_us == 0);
    require(state.layers[0].hot_join_time_us == 0);
    require(state.layers[0].cold_join_time_us == 0);
    require(state.layers[0].parallel_fallbacks == 0);
    require(state.layers[0].parallel_split_debug_samples == 0);
    require(state.layers[0].parallel_split_debug_hot_begin == -1);
    require(state.layers[0].parallel_split_debug_join_backend == -1);
    require(state.layers[0].gate_time_us == 0);
    require(state.layers[0].up_time_us == 0);
    require(state.layers[0].down_time_us == 0);
    require(state.layers[0].experts[1] == 0);
    require(state.layers[0].hot_experts[2] == 0);
    require(state.layers[0].cold_experts[0] == 0);
    require(state.updates == 0);
    require(state.overflow_resets == 1);
    require(state.last_callback_us == 0);
    require(!state.has_data_locked());
}

static void test_saturating_add_resets_on_overflow() {
    llama_moe_layer_perf_state state;
    state.ensure_shape_locked(1, 1, 1);
    state.layers[0].calls = std::numeric_limits<uint64_t>::max();
    state.layers[0].expert_hits_total = 7;
    state.updates = 5;

    state.add_locked(state.layers[0].calls, 1);

    require(state.layers[0].calls == 1);
    require(state.layers[0].expert_hits_total == 0);
    require(state.updates == 0);
    require(state.overflow_resets == 1);
}

static void test_begin_resets_update_counter_overflow() {
    llama_moe_layer_perf_state state;
    state.ensure_shape_locked(1, 1, 1);
    state.layers[0].calls = 42;
    state.updates = std::numeric_limits<uint64_t>::max();

    state.begin_locked(1, 1, 1);

    require(state.active);
    require(state.updates == 1);
    require(state.overflow_resets == 1);
    require(state.layers[0].calls == 0);
}

int main() {
    test_ensure_shape_preserves_existing_counts();
    test_begin_and_end_window();
    test_add_expert_counts_ignore_invalid_ids();
    test_reset_clears_counters_and_preserves_shape();
    test_saturating_add_resets_on_overflow();
    test_begin_resets_update_counter_overflow();
    return 0;
}
