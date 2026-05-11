#include "../src/llama-moe-hot-cache.h"

#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition) {
    if (!condition) {
        throw std::runtime_error("test assertion failed");
    }
}

static void test_parse_and_sort() {
    const std::string json = R"({
        "enabled": true,
        "schema": "llama.cpp.moe_layer_perf.v1",
        "n_expert": 4,
        "n_expert_used": 2,
        "layers": [
            {"layer": 2, "experts": [[3, 5], [1, 7], [0, 0]]},
            {"layer": 1, "experts": [[2, 7], [0, 1]]}
        ]
    })";

    const auto entries = llama_moe_hot_cache_parse_perf_json(json);
    require(entries.size() == 4);
    require(entries[0].layer == 1 && entries[0].expert == 2 && entries[0].hit_count == 7);
    require(entries[1].layer == 2 && entries[1].expert == 1 && entries[1].hit_count == 7);
    require(entries[2].layer == 2 && entries[2].expert == 3 && entries[2].hit_count == 5);
    require(entries[3].layer == 1 && entries[3].expert == 0 && entries[3].hit_count == 1);
}

static void test_select_budget() {
    const std::vector<llama_moe_hot_cache_entry> observed = {
        { 0, 0, 10 },
        { 0, 1, 9  },
        { 1, 0, 8  },
        { 1, 1, 7  },
    };

    const std::vector<llama_moe_hot_cache_expert_size> sizes = {
        { 0, 0, 40 },
        { 0, 1, 70 },
        { 1, 0, 20 },
        { 1, 1, 10 },
    };

    const auto plan = llama_moe_hot_cache_select(observed, sizes, 80);
    require(plan.used_bytes == 70);
    require(plan.selected.size() == 3);
    require(plan.selected[0].layer == 0 && plan.selected[0].expert == 0);
    require(plan.selected[1].layer == 1 && plan.selected[1].expert == 0);
    require(plan.selected[2].layer == 1 && plan.selected[2].expert == 1);
}

static void test_bad_schema() {
    bool threw = false;
    try {
        (void) llama_moe_hot_cache_parse_perf_json(R"({"schema":"wrong","layers":[]})");
    } catch (const std::runtime_error &) {
        threw = true;
    }
    require(threw);
}

int main() {
    test_parse_and_sort();
    test_select_budget();
    test_bad_schema();
    return 0;
}
