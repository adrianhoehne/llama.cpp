#include "../src/moe-hot-cache/llama-moe-hot-cache-planner.h"

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

    require(llama_moe_hot_cache_tensor_expert_bytes(nullptr) == 0);
    require(llama_moe_hot_cache_tensor_expert_bytes(tensor) == ggml_nbytes(tensor)/3);
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

int main() {
    test_tensor_expert_bytes_splits_by_expert_dimension();
    test_select_accounts_for_one_dummy_expert_per_active_layer();
    test_select_ignores_observed_entries_without_size();
    test_select_skips_too_expensive_entries_and_continues();
    test_select_rejects_overflowing_layer_dummy_cost();
    return 0;
}
