#include "../src/llama-moe-hot-cache.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
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

static void test_default_weighting_is_flat() {
    scoped_env_var weighting("LLAMA_MOE_HOT_CACHE_WEIGHTING", nullptr);
    scoped_env_var qwen_weighting("LLAMA_MOE_HOT_CACHE_QWEN_WEIGHTING", nullptr);

    const auto config = llama_moe_hot_cache_weighting::default_config();
    require(config.mode == llama_moe_hot_cache_weighting_mode::flat);
}

static bool hot_dummy_padding_enabled() {
    const char * env = std::getenv("LLAMA_MOE_HOT_CACHE_HOT_DUMMY_PADDING");
    return env == nullptr || env[0] == '\0' ||
           (std::strcmp(env, "0") != 0 && std::strcmp(env, "off") != 0 && std::strcmp(env, "false") != 0);
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

static void test_parse_branch_counts_and_layer_weight() {
    const std::string json = R"({
        "enabled": true,
        "schema": "llama.cpp.moe_layer_perf.v1",
        "n_expert": 4,
        "n_expert_used": 2,
        "layers": [
            {
                "layer": 0,
                "experts": [[0, 100]],
                "hot_experts": [[1, 10]],
                "cold_experts": [[2, 10]],
                "calls": 10,
                "cold_slots_total": 10,
                "parallel_join_wait_time_per_call_us": 20.0
            },
            {
                "layer": 1,
                "hot_experts": [[3, 12]],
                "calls": 10,
                "cold_slots_total": 10,
                "parallel_join_wait_time_per_call_us": 10.0
            }
        ]
    })";

    llama_moe_hot_cache_qwen35moe_weighting_config config;
    config.mode = llama_moe_hot_cache_weighting_mode::pressure;

    const auto entries = llama_moe_hot_cache_qwen35moe_weighting::score_observations(
            llama_moe_hot_cache_parse_perf_json_observations(json), config);
    require(entries.size() == 3);
    require(entries[0].layer == 0 && entries[0].expert == 1 && entries[0].hit_count == 12);
    require(entries[1].layer == 0 && entries[1].expert == 2 && entries[1].hit_count == 12);
    require(entries[2].layer == 1 && entries[2].expert == 3 && entries[2].hit_count == 11);
}

static void test_qwen_layer_pressure_uses_total_wait() {
    const std::string json = R"({
        "enabled": true,
        "schema": "llama.cpp.moe_layer_perf.v1",
        "n_expert": 4,
        "n_expert_used": 2,
        "layers": [
            {
                "layer": 0,
                "cold_experts": [[1, 10]],
                "cold_slots_per_call": 10.0,
                "parallel_join_wait_time_per_call_us": 20.0
            },
            {
                "layer": 1,
                "cold_experts": [[2, 10]],
                "cold_slots_per_call": 5.0,
                "parallel_join_wait_time_per_call_us": 10.0
            }
        ]
    })";

    llama_moe_hot_cache_qwen35moe_weighting_config config;
    config.mode = llama_moe_hot_cache_weighting_mode::pressure;

    const auto entries = llama_moe_hot_cache_qwen35moe_weighting::score_observations(
            llama_moe_hot_cache_parse_perf_json_observations(json), config);
    require(entries.size() == 2);
    require(entries[0].layer == 0 && entries[0].expert == 1 && entries[0].hit_count == 12);
    require(entries[1].layer == 1 && entries[1].expert == 2 && entries[1].hit_count == 8);
}

static void test_gemma4_layer_pressure_uses_total_wait() {
    const std::string json = R"({
        "enabled": true,
        "schema": "llama.cpp.moe_layer_perf.v1",
        "n_expert": 4,
        "n_expert_used": 2,
        "layers": [
            {
                "layer": 0,
                "cold_experts": [[1, 10]],
                "cold_slots_per_call": 10.0,
                "parallel_join_wait_time_per_call_us": 20.0
            },
            {
                "layer": 1,
                "cold_experts": [[2, 10]],
                "cold_slots_per_call": 5.0,
                "parallel_join_wait_time_per_call_us": 10.0
            }
        ]
    })";

    scoped_env_var weighting("LLAMA_MOE_HOT_CACHE_WEIGHTING", "pressure");

    const auto entries = llama_moe_hot_cache_gemma4_weighting::score_observations(
            llama_moe_hot_cache_parse_perf_json_observations(json));
    require(entries.size() == 2);
    require(entries[0].layer == 0 && entries[0].expert == 1 && entries[0].hit_count == 12);
    require(entries[1].layer == 1 && entries[1].expert == 2 && entries[1].hit_count == 8);
}

static void test_flat_weighting_spreads_budget_over_layers() {
    const std::string json = R"({
        "enabled": true,
        "schema": "llama.cpp.moe_layer_perf.v1",
        "n_expert": 4,
        "n_expert_used": 2,
        "layers": [
            { "layer": 0, "experts": [[0, 100], [1, 90], [2, 80]] },
            { "layer": 1, "experts": [[0, 50],  [1, 40], [2, 30]] },
            { "layer": 2, "experts": [[0, 20],  [1, 10], [2, 5]] }
        ]
    })";

    llama_moe_hot_cache_weighting_config config;
    config.mode = llama_moe_hot_cache_weighting_mode::flat;
    config.layer_curve = 1.0;

    const auto observations = llama_moe_hot_cache_parse_perf_json_observations(json);
    const auto entries = llama_moe_hot_cache_weighting::score_observations(observations, config);
    require(entries.size() == 9);
    require(entries[0].layer == 0 && entries[0].expert == 0);
    require(entries[1].layer == 1 && entries[1].expert == 0);
    require(entries[2].layer == 2 && entries[2].expert == 0);
    require(entries[3].layer == 0 && entries[3].expert == 1);
    require(entries[4].layer == 1 && entries[4].expert == 1);
    require(entries[5].layer == 2 && entries[5].expert == 1);

    std::vector<llama_moe_hot_cache_expert_size> sizes;
    sizes.reserve(entries.size());
    for (const auto & entry : entries) {
        sizes.push_back({ entry.layer, entry.expert, 1 });
    }

    const auto plan = llama_moe_hot_cache_select(entries, sizes, 9);
    require(plan.selected.size() == 6);

    int per_layer[3] = {};
    for (const auto & selected : plan.selected) {
        require(selected.layer < 3);
        per_layer[selected.layer]++;
    }

    require(per_layer[0] == 2);
    require(per_layer[1] == 2);
    require(per_layer[2] == 2);
}

static void test_parse_raw_opt_schema() {
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

    const auto entries = llama_moe_hot_cache_parse_perf_json(json);
    require(entries.size() == 6);
    require(entries[0].layer == 1 && entries[0].expert == 1 && entries[0].hit_count == 6);
    require(entries[1].layer == 0 && entries[1].expert == 1 && entries[1].hit_count == 5);
    require(entries[2].layer == 0 && entries[2].expert == 3 && entries[2].hit_count == 4);
    require(entries[3].layer == 1 && entries[3].expert == 2 && entries[3].hit_count == 3);
    require(entries[4].layer == 0 && entries[4].expert == 0 && entries[4].hit_count == 2);
    require(entries[5].layer == 1 && entries[5].expert == 0 && entries[5].hit_count == 1);
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
    require(plan.used_bytes == 80);
    require(plan.selected.size() == 1);
    require(plan.selected[0].layer == 0 && plan.selected[0].expert == 0);
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

static ggml_context_ptr make_ctx() {
    ggml_init_params params = {
        /*.mem_size   =*/ 16*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    return ggml_context_ptr(ggml_init(params));
}

static float get_worklist_field(const ggml_tensor * packed, int32_t field, int32_t slot) {
    const char * row = (const char *) packed->data + field*packed->nb[1];
    return *(const float *)(row + slot*packed->nb[0]);
}

static void set_selected(ggml_tensor * selected, int32_t iex, int32_t token, int32_t value) {
    *(int32_t *) ((char *) selected->data + iex*selected->nb[0] + token*selected->nb[1]) = value;
}

static void set_weight(ggml_tensor * weights, int32_t iex, int32_t token, float value) {
    *(float *) ((char *) weights->data + iex*weights->nb[1] + token*weights->nb[2]) = value;
}

static void set_logit(ggml_tensor * logits, int32_t expert, int32_t token, float value) {
    *(float *) ((char *) logits->data + expert*logits->nb[0] + token*logits->nb[1]) = value;
}

static void require_close(float actual, float expected) {
    require(std::fabs(actual - expected) < 1e-5f);
}

static void test_build_worklist_mixed() {
    auto ctx = make_ctx();
    require(ctx != nullptr);

    const int32_t n_expert_used = 2;
    const int32_t n_tokens = 3;
    const int32_t capacity = n_expert_used*n_tokens;

    ggml_tensor * selected = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_expert_used, n_tokens);
    ggml_tensor * weights = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, n_expert_used, n_tokens);
    ggml_tensor * packed = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, capacity, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COUNT);

    set_selected(selected, 0, 0, 1);
    set_selected(selected, 1, 0, 3);
    set_selected(selected, 0, 1, 2);
    set_selected(selected, 1, 1, 1);
    set_selected(selected, 0, 2, 0);
    set_selected(selected, 1, 2, 2);

    set_weight(weights, 0, 0, 0.11f);
    set_weight(weights, 1, 0, 0.12f);
    set_weight(weights, 0, 1, 0.21f);
    set_weight(weights, 1, 1, 0.22f);
    set_weight(weights, 0, 2, 0.31f);
    set_weight(weights, 1, 2, 0.32f);

    llama_moe_hot_cache_layer layer;
    layer.n_expert = 4;
    layer.n_hot = 2;
    layer.hot_id_map_host = { -1, 0, -1, 1 };

    llama_moe_hot_cache_build_worklist(packed, selected, weights, layer, 0, 1);

    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID, 0) == 0.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT, 0) == 0.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID, 0) == 0.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT, 0) == 0.11f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, 0) == 1.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID, 1) == 1.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT, 1) == 1.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID, 1) == 0.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT, 1) == 0.12f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, 1) == 3.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID, 2) == 0.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT, 2) == 3.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID, 2) == 1.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT, 2) == 0.22f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, 2) == 1.0f);

    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID, 0) == 2.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT, 0) == 2.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, 0) == 1.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT, 0) == 0.21f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID, 1) == 0.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT, 1) == 4.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, 1) == 2.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT, 1) == 0.31f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID, 2) == 2.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT, 2) == 5.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, 2) == 2.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT, 2) == 0.32f);

    for (int32_t i = 3; i < capacity; ++i) {
        const float expected_hot_padding = hot_dummy_padding_enabled() ? float(layer.n_hot) : -1.0f;
        require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID, i) == expected_hot_padding);
        require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT, i) == float(capacity));
        require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, i) == -1.0f);
        require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID, i) == -1.0f);
        require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT, i) == float(capacity));
    }
}

static void test_build_worklist_all_hot_or_cold() {
    auto ctx = make_ctx();
    require(ctx != nullptr);

    const int32_t n_expert_used = 2;
    const int32_t n_tokens = 2;
    const int32_t capacity = n_expert_used*n_tokens;

    ggml_tensor * selected = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_expert_used, n_tokens);
    ggml_tensor * weights = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, n_expert_used, n_tokens);
    ggml_tensor * packed = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, capacity, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COUNT);

    for (int32_t token = 0; token < n_tokens; ++token) {
        for (int32_t iex = 0; iex < n_expert_used; ++iex) {
            set_selected(selected, iex, token, iex);
            set_weight(weights, iex, token, 0.5f + 0.1f*float(token*n_expert_used + iex));
        }
    }

    llama_moe_hot_cache_layer all_hot;
    all_hot.n_expert = 2;
    all_hot.n_hot = 2;
    all_hot.hot_id_map_host = { 0, 1 };
    llama_moe_hot_cache_build_worklist(packed, selected, weights, all_hot, 0, 1);
    for (int32_t i = 0; i < capacity; ++i) {
        require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID, i) >= 0.0f);
        require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, i) >= 0.0f);
        require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID, i) == -1.0f);
    }

    llama_moe_hot_cache_layer all_cold;
    all_cold.n_expert = 2;
    all_cold.n_hot = 0;
    all_cold.hot_id_map_host = { -1, -1 };
    llama_moe_hot_cache_build_worklist(packed, selected, weights, all_cold, 0, 1);
    for (int32_t i = 0; i < capacity; ++i) {
        require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID, i) == -1.0f);
        require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, i) == -1.0f);
        require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID, i) >= 0.0f);
    }
}

static void test_build_worklist_from_logits() {
    auto ctx = make_ctx();
    require(ctx != nullptr);

    const int32_t n_expert_used = 2;
    const int32_t n_tokens = 1;
    const int32_t capacity = n_expert_used*n_tokens;

    ggml_tensor * logits = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, n_tokens);
    ggml_tensor * packed = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, capacity, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COUNT);

    set_logit(logits, 0, 0, 0.0f);
    set_logit(logits, 1, 0, 2.0f);
    set_logit(logits, 2, 0, 1.0f);
    set_logit(logits, 3, 0, 3.0f);

    llama_moe_hot_cache_layer layer;
    layer.n_expert = 4;
    layer.n_hot = 1;
    layer.expert_weights_scale = 2.0f;
    layer.hot_id_map_host = { -1, -1, -1, 0 };

    llama_moe_hot_cache_build_worklist_from_logits(packed, logits, layer, 0, 1);

    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID, 0) == 0.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT, 0) == 0.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID, 0) == 0.0f);
    require_close(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT, 0), 1.4621172f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, 0) == 3.0f);

    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID, 0) == 1.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT, 0) == 1.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, 0) == 0.0f);
    require_close(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT, 0), 0.5378828f);

    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_COUNT, 0) == 1.0f);
    require(get_worklist_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_COUNT, 0) == 1.0f);
}

int main() {
    test_default_weighting_is_flat();
    test_parse_and_sort();
    test_parse_branch_counts_and_layer_weight();
    test_qwen_layer_pressure_uses_total_wait();
    test_gemma4_layer_pressure_uses_total_wait();
    test_flat_weighting_spreads_budget_over_layers();
    test_parse_raw_opt_schema();
    test_select_budget();
    test_bad_schema();
    test_build_worklist_mixed();
    test_build_worklist_all_hot_or_cold();
    test_build_worklist_from_logits();
    return 0;
}
