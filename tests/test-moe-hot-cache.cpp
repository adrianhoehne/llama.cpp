#include "../src/moe-hot-cache/llama-moe-hot-cache.h"

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

static bool hot_dummy_padding_enabled() {
    const char * env = std::getenv("LLAMA_MOE_HOT_CACHE_HOT_DUMMY_PADDING");
    return env == nullptr || env[0] == '\0' ||
           (std::strcmp(env, "0") != 0 && std::strcmp(env, "off") != 0 && std::strcmp(env, "false") != 0);
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
    test_build_worklist_mixed();
    test_build_worklist_all_hot_or_cold();
    test_build_worklist_from_logits();
    return 0;
}
