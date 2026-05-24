#include "../src/moe-hot-cache/llama-moe-hot-cache-budget.h"
#include "../src/moe-hot-cache/llama-moe-hot-cache-common.h"

#include <limits>
#include <stdexcept>
#include <string>

static void require_impl(bool condition, int line) {
    if (!condition) {
        throw std::runtime_error("test assertion failed at line " + std::to_string(line));
    }
}

#define require(condition) require_impl((condition), __LINE__)

static void test_compute_auto_budget_rounds_down_to_mib() {
    const size_t free_bytes = 4096*LLAMA_MOE_HOT_CACHE_MIB + LLAMA_MOE_HOT_CACHE_MIB/2;
    const size_t kv_reserve = 1024*LLAMA_MOE_HOT_CACHE_MIB;

    const size_t budget = llama_moe_hot_cache_compute_auto_budget_bytes(
            free_bytes,
            kv_reserve,
            512);

    require(budget == 2560*LLAMA_MOE_HOT_CACHE_MIB);
}

static void test_compute_auto_budget_without_reserves() {
    const size_t free_bytes = 5*LLAMA_MOE_HOT_CACHE_MIB + 123;

    const size_t budget = llama_moe_hot_cache_compute_auto_budget_bytes(
            free_bytes,
            0,
            0);

    require(budget == 5*LLAMA_MOE_HOT_CACHE_MIB);
}

static void test_compute_auto_budget_returns_zero_when_reserved_exhausts_free_memory() {
    require(llama_moe_hot_cache_compute_auto_budget_bytes(
            1536*LLAMA_MOE_HOT_CACHE_MIB,
            1024*LLAMA_MOE_HOT_CACHE_MIB,
            512) == 0);

    require(llama_moe_hot_cache_compute_auto_budget_bytes(
            1535*LLAMA_MOE_HOT_CACHE_MIB,
            1024*LLAMA_MOE_HOT_CACHE_MIB,
            512) == 0);
}

static void test_compute_auto_budget_saturates_reserve_overflow() {
    const size_t huge_mib = std::numeric_limits<size_t>::max() / LLAMA_MOE_HOT_CACHE_MIB + 1;

    const size_t budget = llama_moe_hot_cache_compute_auto_budget_bytes(
            4096*LLAMA_MOE_HOT_CACHE_MIB,
            1024*LLAMA_MOE_HOT_CACHE_MIB,
            huge_mib);

    require(budget == 0);
}

int main() {
    test_compute_auto_budget_rounds_down_to_mib();
    test_compute_auto_budget_without_reserves();
    test_compute_auto_budget_returns_zero_when_reserved_exhausts_free_memory();
    test_compute_auto_budget_saturates_reserve_overflow();
    return 0;
}
