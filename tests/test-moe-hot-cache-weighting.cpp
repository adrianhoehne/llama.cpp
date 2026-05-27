#include "../src/moe-hot-cache/llama-moe-hot-cache.h"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static void require_impl(bool condition, int line) {
    if (!condition) {
        throw std::runtime_error("test assertion failed at line " + std::to_string(line));
    }
}

#define require(condition) require_impl((condition), __LINE__)

static void set_env_var(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
#endif
}

class scoped_env_var {
public:
    scoped_env_var(const char * name, const char * value) : name(name) {
        const char * current = std::getenv(name);
        if (current != nullptr) {
            had_value = true;
            old_value = current;
        }
        set_env_var(name, value);
    }

    ~scoped_env_var() {
        set_env_var(name, had_value ? old_value.c_str() : nullptr);
    }

private:
    const char * name;
    bool had_value = false;
    std::string old_value;
};

static llama_moe_hot_cache_layer_observation raw_layer(
        uint32_t layer,
        const std::vector<std::pair<uint32_t, uint64_t>> & experts) {
    llama_moe_hot_cache_layer_observation result;
    result.layer = layer;
    for (const auto & [expert, hits] : experts) {
        result.experts.push_back({ expert, 0, 0, hits });
    }
    return result;
}

static llama_moe_hot_cache_layer_observation branch_layer(
        uint32_t layer,
        double join_wait_us,
        double cold_slots_per_call,
        const std::vector<std::pair<uint32_t, uint64_t>> & hot_experts,
        const std::vector<std::pair<uint32_t, uint64_t>> & cold_experts) {
    llama_moe_hot_cache_layer_observation result;
    result.layer = layer;
    result.has_branch_counts = true;
    result.parallel_join_wait_time_per_call_us = join_wait_us;
    result.cold_slots_per_call = cold_slots_per_call;
    result.wait_per_cold_slot_us = cold_slots_per_call > 0.0 ? join_wait_us / cold_slots_per_call : 0.0;

    for (const auto & [expert, hits] : hot_experts) {
        result.experts.push_back({ expert, hits, 0, 0 });
    }
    for (const auto & [expert, hits] : cold_experts) {
        result.experts.push_back({ expert, 0, hits, 0 });
    }
    return result;
}

static void test_default_weighting_is_flat() {
    scoped_env_var weighting("LLAMA_MOE_HOT_CACHE_WEIGHTING", nullptr);
    scoped_env_var qwen_weighting("LLAMA_MOE_HOT_CACHE_QWEN_WEIGHTING", nullptr);

    const auto config = llama_moe_hot_cache_weighting::default_config();
    require(config.mode == llama_moe_hot_cache_weighting_mode::flat);
}

static void test_parse_modes_and_names() {
    llama_moe_hot_cache_weighting_mode mode = llama_moe_hot_cache_weighting_mode::flat;
    require(llama_moe_hot_cache_weighting::parse_mode("pressure", mode));
    require(mode == llama_moe_hot_cache_weighting_mode::pressure);
    require(std::string(llama_moe_hot_cache_weighting::mode_name(mode)) == "pressure");

    require(llama_moe_hot_cache_weighting::parse_mode("smooth-pressure", mode));
    require(mode == llama_moe_hot_cache_weighting_mode::smooth_pressure);
    require(std::string(llama_moe_hot_cache_weighting::mode_name(mode)) == "smooth");

    require(llama_moe_hot_cache_weighting::parse_mode("decode-time", mode));
    require(mode == llama_moe_hot_cache_weighting_mode::time);
    require(std::string(llama_moe_hot_cache_weighting::mode_name(mode)) == "time");

    require(llama_moe_hot_cache_weighting::parse_mode("layer-rank", mode));
    require(mode == llama_moe_hot_cache_weighting_mode::balanced);
    require(std::string(llama_moe_hot_cache_weighting::mode_name(mode)) == "balanced");

    require(llama_moe_hot_cache_weighting::parse_mode("flat", mode));
    require(mode == llama_moe_hot_cache_weighting_mode::flat);
    require(std::string(llama_moe_hot_cache_weighting::mode_name(mode)) == "flat");

    require(!llama_moe_hot_cache_weighting::parse_mode("unknown", mode));
}

static void test_pressure_weighting_uses_total_layer_wait() {
    const std::vector<llama_moe_hot_cache_layer_observation> observations = {
        branch_layer(0, 20.0, 10.0, {}, {{1, 10}}),
        branch_layer(1, 10.0, 5.0,  {}, {{2, 11}}),
    };

    llama_moe_hot_cache_weighting_config config;
    config.mode = llama_moe_hot_cache_weighting_mode::pressure;

    const auto entries = llama_moe_hot_cache_weighting::score_observations(observations, config);
    require(entries.size() == 2);
    require(entries[0].layer == 0 && entries[0].expert == 1 && entries[0].hit_count == 12);
    require(entries[1].layer == 1 && entries[1].expert == 2 && entries[1].hit_count == 9);
}

static void test_pressure_weighting_applies_hot_sticky_bonus() {
    const std::vector<llama_moe_hot_cache_layer_observation> observations = {
        branch_layer(0, 20.0, 1.0, {{1, 10}}, {{2, 10}}),
        branch_layer(1, 10.0, 1.0, {{3, 12}}, {}),
    };

    llama_moe_hot_cache_weighting_config config;
    config.mode = llama_moe_hot_cache_weighting_mode::pressure;

    const auto entries = llama_moe_hot_cache_weighting::score_observations(observations, config);
    require(entries.size() == 3);
    require(entries[0].layer == 0 && entries[0].expert == 1 && entries[0].hit_count == 12);
    require(entries[1].layer == 0 && entries[1].expert == 2 && entries[1].hit_count == 12);
    require(entries[2].layer == 1 && entries[2].expert == 3 && entries[2].hit_count == 11);
}

static void test_flat_weighting_interleaves_layers() {
    const std::vector<llama_moe_hot_cache_layer_observation> observations = {
        raw_layer(0, {{0, 100}, {1, 90}, {2, 80}}),
        raw_layer(1, {{0, 50},  {1, 40}, {2, 30}}),
        raw_layer(2, {{0, 20},  {1, 10}, {2, 5}}),
    };

    llama_moe_hot_cache_weighting_config config;
    config.mode = llama_moe_hot_cache_weighting_mode::flat;
    config.layer_curve = 1.0;

    const auto entries = llama_moe_hot_cache_weighting::score_observations(observations, config);
    require(entries.size() == 9);
    require(entries[0].layer == 0 && entries[0].expert == 0);
    require(entries[1].layer == 1 && entries[1].expert == 0);
    require(entries[2].layer == 2 && entries[2].expert == 0);
    require(entries[3].layer == 0 && entries[3].expert == 1);
    require(entries[4].layer == 1 && entries[4].expert == 1);
    require(entries[5].layer == 2 && entries[5].expert == 1);
}

static void test_qwen_wrapper_delegates_to_generic_weighting() {
    const std::vector<llama_moe_hot_cache_layer_observation> observations = {
        raw_layer(0, {{1, 20}, {2, 10}}),
        raw_layer(1, {{3, 30}}),
    };

    llama_moe_hot_cache_qwen35moe_weighting_config config;
    config.mode = llama_moe_hot_cache_weighting_mode::balanced;

    const auto generic = llama_moe_hot_cache_weighting::score_observations(observations, config);
    const auto qwen = llama_moe_hot_cache_qwen35moe_weighting::score_observations(observations, config);
    require(generic.size() == qwen.size());
    for (size_t i = 0; i < generic.size(); ++i) {
        require(generic[i].layer == qwen[i].layer);
        require(generic[i].expert == qwen[i].expert);
        require(generic[i].hit_count == qwen[i].hit_count);
    }
}

int main() {
    test_default_weighting_is_flat();
    test_parse_modes_and_names();
    test_pressure_weighting_uses_total_layer_wait();
    test_pressure_weighting_applies_hot_sticky_bonus();
    test_flat_weighting_interleaves_layers();
    test_qwen_wrapper_delegates_to_generic_weighting();
    return 0;
}
