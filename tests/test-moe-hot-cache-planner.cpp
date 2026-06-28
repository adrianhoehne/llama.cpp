#include "../src/moe-hot-cache/llama-moe-hot-cache-planner.h"

#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void require_impl(bool condition, int line) {
    if (!condition) {
        throw std::runtime_error("test assertion failed at line " + std::to_string(line));
    }
}

#define require(condition) require_impl((condition), __LINE__)

static bool has_selected_expert(const llama_moe_hot_cache_multi_plan & plan, uint32_t layer, uint32_t expert) {
    for (const auto & lane : plan.lanes) {
        for (const auto & selected : lane.selected) {
            if (selected.layer == layer && selected.expert == expert) {
                return true;
            }
        }
    }

    return false;
}

static bool has_selected_expert(const llama_moe_hot_cache_plan & plan, uint32_t layer, uint32_t expert) {
    for (const auto & selected : plan.selected) {
        if (selected.layer == layer && selected.expert == expert) {
            return true;
        }
    }

    return false;
}

static size_t selected_count_for_lane_layer(
        const llama_moe_hot_cache_multi_plan & plan,
        size_t lane,
        uint32_t layer) {
    size_t result = 0;
    for (const auto & selected : plan.lanes.at(lane).selected) {
        if (selected.layer == layer) {
            result++;
        }
    }
    return result;
}

static ggml_context_ptr make_ctx() {
    ggml_init_params params = {
        /*.mem_size   =*/ 64*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    return ggml_context_ptr(ggml_init(params));
}

static void test_tensor_expert_bytes_splits_by_expert_dimension() {
    auto ctx = make_ctx();
    require(ctx != nullptr);

    ggml_tensor * tensor = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 4, 5, 3);
    ggml_tensor * bias = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 3);

    require(llama_moe_hot_cache_tensor_expert_bytes(nullptr) == 0);
    require(llama_moe_hot_cache_tensor_expert_bytes(tensor) == ggml_nbytes(tensor)/3);
    require(llama_moe_hot_cache_tensor_expert_bias_bytes(nullptr) == 0);
    require(llama_moe_hot_cache_tensor_expert_bias_bytes(bias) == ggml_nbytes(bias)/3);
}

static void test_select_accounts_for_one_dummy_expert_per_active_layer() {
    const std::vector<llama_moe_hot_cache_entry> observed = {
        { 0, 0, 100 },
        { 0, 1, 90  },
        { 1, 0, 80  },
    };

    const std::vector<llama_moe_hot_cache_expert_size> sizes = {
        { 0, 0, 10 },
        { 0, 1, 10 },
        { 1, 0, 10 },
    };

    const auto plan = llama_moe_hot_cache_select(observed, sizes, 30);

    require(plan.budget_bytes == 30);
    require(plan.used_bytes == 30);
    require(plan.selected.size() == 2);
    require(plan.selected[0].layer == 0 && plan.selected[0].expert == 0);
    require(plan.selected[1].layer == 0 && plan.selected[1].expert == 1);
}

static void test_select_ignores_observed_entries_without_size() {
    const std::vector<llama_moe_hot_cache_entry> observed = {
        { 0, 0, 100 },
        { 0, 1, 90  },
    };

    const std::vector<llama_moe_hot_cache_expert_size> sizes = {
        { 0, 1, 10 },
    };

    const auto plan = llama_moe_hot_cache_select(observed, sizes, 20);

    require(plan.used_bytes == 20);
    require(plan.selected.size() == 1);
    require(plan.selected[0].layer == 0 && plan.selected[0].expert == 1);
}

static void test_select_skips_too_expensive_entries_and_continues() {
    const std::vector<llama_moe_hot_cache_entry> observed = {
        { 0, 0, 100 },
        { 0, 1, 90  },
    };

    const std::vector<llama_moe_hot_cache_expert_size> sizes = {
        { 0, 0, 100 },
        { 0, 1, 10  },
    };

    const auto plan = llama_moe_hot_cache_select(observed, sizes, 20);

    require(plan.used_bytes == 20);
    require(plan.selected.size() == 1);
    require(plan.selected[0].layer == 0 && plan.selected[0].expert == 1);
}

static void test_select_rejects_overflowing_layer_dummy_cost() {
    const std::vector<llama_moe_hot_cache_entry> observed = {
        { 0, 0, 100 },
    };

    const std::vector<llama_moe_hot_cache_expert_size> sizes = {
        { 0, 0, std::numeric_limits<size_t>::max()/2 + 1 },
    };

    const auto plan = llama_moe_hot_cache_select(observed, sizes, std::numeric_limits<size_t>::max());

    require(plan.used_bytes == 0);
    require(plan.selected.empty());
}

static void test_select_multi_warm_fills_lanes_without_duplicates() {
    unsetenv("LLAMA_MOE_HOT_CACHE_FILL_RANDOM");

    const std::vector<llama_moe_hot_cache_entry> observed = {
        { 0, 0, 100 },
        { 0, 1, 90  },
        { 0, 2, 80  },
        { 0, 3, 70  },
    };

    const std::vector<llama_moe_hot_cache_expert_size> sizes = {
        { 0, 0, 10 },
        { 0, 1, 10 },
        { 0, 2, 10 },
        { 0, 3, 10 },
    };

    const auto plan = llama_moe_hot_cache_select_multi(
            observed,
            sizes,
            { 30, 20 },
            llama_moe_hot_cache_device_strategy::warm);

    require(plan.lanes.size() == 2);
    require(plan.selected_count() == 3);
    require(plan.used_bytes() == 50);
    require(plan.budget_bytes() == 50);

    require(plan.lanes[0].selected.size() == 2);
    require(plan.lanes[0].selected[0].expert == 0);
    require(plan.lanes[0].selected[1].expert == 1);
    require(plan.lanes[1].selected.size() == 1);
    require(plan.lanes[1].selected[0].expert == 2);
}

static void test_select_multi_hot_even_balances_same_layer() {
    unsetenv("LLAMA_MOE_HOT_CACHE_FILL_RANDOM");

    const std::vector<llama_moe_hot_cache_entry> observed = {
        { 2, 0, 100 },
        { 2, 1, 90  },
        { 2, 2, 80  },
        { 2, 3, 70  },
    };

    const std::vector<llama_moe_hot_cache_expert_size> sizes = {
        { 2, 0, 10 },
        { 2, 1, 10 },
        { 2, 2, 10 },
        { 2, 3, 10 },
    };

    const auto plan = llama_moe_hot_cache_select_multi(
            observed,
            sizes,
            { 40, 40 },
            llama_moe_hot_cache_device_strategy::hot_even);

    require(plan.lanes.size() == 2);
    require(plan.selected_count() == 4);
    require(plan.lanes[0].selected.size() == 2);
    require(plan.lanes[1].selected.size() == 2);
    require(plan.lanes[0].selected[0].expert == 0);
    require(plan.lanes[1].selected[0].expert == 1);
    require(plan.lanes[0].selected[1].expert == 2);
    require(plan.lanes[1].selected[1].expert == 3);
}

static void test_select_multi_even_split_uses_budget_weighted_layer_bands() {
    unsetenv("LLAMA_MOE_HOT_CACHE_FILL_RANDOM");

    const std::vector<llama_moe_hot_cache_entry> observed = {
        { 0, 0, 800 }, { 1, 0, 700 }, { 2, 0, 600 }, { 3, 0, 500 },
        { 0, 1, 400 }, { 1, 1, 300 }, { 2, 1, 200 }, { 3, 1, 100 },
    };

    const std::vector<llama_moe_hot_cache_expert_size> sizes = {
        { 0, 0, 10 }, { 0, 1, 10 },
        { 1, 0, 10 }, { 1, 1, 10 },
        { 2, 0, 10 }, { 2, 1, 10 },
        { 3, 0, 10 }, { 3, 1, 10 },
    };

    const auto plan = llama_moe_hot_cache_select_multi(
            observed,
            sizes,
            { 30, 60, 30 },
            llama_moe_hot_cache_device_strategy::even_split);

    require(plan.lanes.size() == 3);
    require(plan.selected_count() == 8);
    require(plan.lanes[0].used_bytes == 30);
    require(plan.lanes[1].used_bytes == 60);
    require(plan.lanes[2].used_bytes == 30);

    require(selected_count_for_lane_layer(plan, 0, 0) == 2);
    require(selected_count_for_lane_layer(plan, 0, 1) == 0);
    require(selected_count_for_lane_layer(plan, 0, 2) == 0);
    require(selected_count_for_lane_layer(plan, 0, 3) == 0);

    require(selected_count_for_lane_layer(plan, 1, 0) == 0);
    require(selected_count_for_lane_layer(plan, 1, 1) == 2);
    require(selected_count_for_lane_layer(plan, 1, 2) == 2);
    require(selected_count_for_lane_layer(plan, 1, 3) == 0);

    require(selected_count_for_lane_layer(plan, 2, 0) == 0);
    require(selected_count_for_lane_layer(plan, 2, 1) == 0);
    require(selected_count_for_lane_layer(plan, 2, 2) == 0);
    require(selected_count_for_lane_layer(plan, 2, 3) == 2);
}

static void test_select_multi_random_fill_uses_unobserved_experts_when_enabled() {
    setenv("LLAMA_MOE_HOT_CACHE_FILL_RANDOM", "1", 1);

    const std::vector<llama_moe_hot_cache_entry> observed = {
        { 0, 0, 100 },
    };

    const std::vector<llama_moe_hot_cache_expert_size> sizes = {
        { 0, 0, 10 },
        { 0, 1, 10 },
        { 0, 2, 10 },
        { 0, 3, 10 },
    };

    const auto plan = llama_moe_hot_cache_select_multi(
            observed,
            sizes,
            { 50 },
            llama_moe_hot_cache_device_strategy::warm);

    unsetenv("LLAMA_MOE_HOT_CACHE_FILL_RANDOM");

    require(plan.selected_count() == 4);
    require(plan.used_bytes() == 50);
    require(has_selected_expert(plan, 0, 0));
    require(has_selected_expert(plan, 0, 1));
    require(has_selected_expert(plan, 0, 2));
    require(has_selected_expert(plan, 0, 3));
}

static void test_select_random_fill_is_enabled_by_default() {
    unsetenv("LLAMA_MOE_HOT_CACHE_FILL_RANDOM");

    const std::vector<llama_moe_hot_cache_entry> observed = {
        { 0, 0, 100 },
    };

    const std::vector<llama_moe_hot_cache_expert_size> sizes = {
        { 0, 0, 10 },
        { 0, 1, 10 },
        { 0, 2, 10 },
        { 0, 3, 10 },
    };

    const auto plan = llama_moe_hot_cache_select(observed, sizes, 50);

    require(plan.selected.size() == 4);
    require(plan.used_bytes == 50);
    require(has_selected_expert(plan, 0, 0));
    require(has_selected_expert(plan, 0, 1));
    require(has_selected_expert(plan, 0, 2));
    require(has_selected_expert(plan, 0, 3));
}

static void test_select_multi_random_fill_is_enabled_by_default() {
    unsetenv("LLAMA_MOE_HOT_CACHE_FILL_RANDOM");

    const std::vector<llama_moe_hot_cache_entry> observed = {
        { 0, 0, 100 },
    };

    const std::vector<llama_moe_hot_cache_expert_size> sizes = {
        { 0, 0, 10 },
        { 0, 1, 10 },
        { 0, 2, 10 },
        { 0, 3, 10 },
    };

    const auto plan = llama_moe_hot_cache_select_multi(
            observed,
            sizes,
            { 50 },
            llama_moe_hot_cache_device_strategy::warm);

    require(plan.selected_count() == 4);
    require(plan.used_bytes() == 50);
    require(has_selected_expert(plan, 0, 0));
    require(has_selected_expert(plan, 0, 1));
    require(has_selected_expert(plan, 0, 2));
    require(has_selected_expert(plan, 0, 3));
}

static void test_select_multi_random_fill_can_be_disabled() {
    setenv("LLAMA_MOE_HOT_CACHE_FILL_RANDOM", "0", 1);

    const std::vector<llama_moe_hot_cache_entry> observed = {
        { 0, 0, 100 },
    };

    const std::vector<llama_moe_hot_cache_expert_size> sizes = {
        { 0, 0, 10 },
        { 0, 1, 10 },
        { 0, 2, 10 },
        { 0, 3, 10 },
    };

    const auto plan = llama_moe_hot_cache_select_multi(
            observed,
            sizes,
            { 50 },
            llama_moe_hot_cache_device_strategy::warm);

    unsetenv("LLAMA_MOE_HOT_CACHE_FILL_RANDOM");

    require(plan.selected_count() == 1);
    require(plan.used_bytes() == 20);
    require(has_selected_expert(plan, 0, 0));
}

static void test_parse_device_strategy() {
    require(llama_moe_hot_cache_parse_device_strategy(nullptr) == llama_moe_hot_cache_device_strategy::warm);
    require(llama_moe_hot_cache_parse_device_strategy("warm") == llama_moe_hot_cache_device_strategy::warm);
    require(llama_moe_hot_cache_parse_device_strategy("hot-even") == llama_moe_hot_cache_device_strategy::hot_even);
    require(llama_moe_hot_cache_parse_device_strategy("even-split") == llama_moe_hot_cache_device_strategy::even_split);

    bool threw = false;
    try {
        (void) llama_moe_hot_cache_parse_device_strategy("invalid");
    } catch (const std::runtime_error &) {
        threw = true;
    }
    require(threw);
}

int main() {
    test_tensor_expert_bytes_splits_by_expert_dimension();
    test_select_accounts_for_one_dummy_expert_per_active_layer();
    test_select_ignores_observed_entries_without_size();
    test_select_skips_too_expensive_entries_and_continues();
    test_select_rejects_overflowing_layer_dummy_cost();
    test_select_multi_warm_fills_lanes_without_duplicates();
    test_select_multi_hot_even_balances_same_layer();
    test_select_multi_even_split_uses_budget_weighted_layer_bands();
    test_select_multi_random_fill_uses_unobserved_experts_when_enabled();
    test_select_random_fill_is_enabled_by_default();
    test_select_multi_random_fill_is_enabled_by_default();
    test_select_multi_random_fill_can_be_disabled();
    test_parse_device_strategy();
    return 0;
}
