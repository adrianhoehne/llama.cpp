#include "../src/moe-hot-cache/llama-moe-hot-cache-builder.h"

#include <cmath>
#include <stdexcept>
#include <string>

static void require_impl(bool condition, int line) {
    if (!condition) {
        throw std::runtime_error("test assertion failed at line " + std::to_string(line));
    }
}

#define require(condition) require_impl((condition), __LINE__)

static void require_close(double actual, double expected) {
    require(std::fabs(actual - expected) < 1e-9);
}

static void test_group_selected_by_layer_preserves_order_and_ignores_out_of_range() {
    llama_moe_hot_cache_plan plan;
    plan.selected = {
        { 2, 7, 10 },
        { 0, 1, 10 },
        { 2, 4, 10 },
        { 4, 9, 10 },
        { 1, 3, 10 },
    };

    const auto grouped = llama_moe_hot_cache_group_selected_by_layer(plan, 3);

    require(grouped.size() == 3);
    require(grouped[0].size() == 1);
    require(grouped[0][0] == 1);
    require(grouped[1].size() == 1);
    require(grouped[1][0] == 3);
    require(grouped[2].size() == 2);
    require(grouped[2][0] == 7);
    require(grouped[2][1] == 4);
}

static void test_summarize_selected_layers_counts_active_layers() {
    const std::vector<std::vector<uint32_t>> selected_by_layer = {
        { 1, 2, 3 },
        {},
        { 4 },
        { 5, 6 },
    };

    const auto stats = llama_moe_hot_cache_summarize_selected_layers(selected_by_layer);

    require(stats.active_layers == 3);
    require(stats.total_hot == 6);
    require(stats.min_hot == 1);
    require(stats.max_hot == 3);
    require_close(stats.avg_hot(), 2.0);
}

static void test_summarize_empty_selection_is_zero() {
    const std::vector<std::vector<uint32_t>> selected_by_layer = {
        {},
        {},
    };

    const auto stats = llama_moe_hot_cache_summarize_selected_layers(selected_by_layer);

    require(stats.active_layers == 0);
    require(stats.total_hot == 0);
    require(stats.min_hot == 0);
    require(stats.max_hot == 0);
    require_close(stats.avg_hot(), 0.0);
}

int main() {
    test_group_selected_by_layer_preserves_order_and_ignores_out_of_range();
    test_summarize_selected_layers_counts_active_layers();
    test_summarize_empty_selection_is_zero();
    return 0;
}
