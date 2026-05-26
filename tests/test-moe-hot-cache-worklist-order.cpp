#include "../src/moe-hot-cache/llama-moe-hot-cache-worklist.h"

#include <cmath>
#include <stdexcept>
#include <string>

static void require_impl(bool condition, int line) {
    if (!condition) {
        throw std::runtime_error("test assertion failed at line " + std::to_string(line));
    }
}

#define require(condition) require_impl((condition), __LINE__)

static ggml_context_ptr make_ctx() {
    ggml_init_params params = {
        /*.mem_size   =*/ 32*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    return ggml_context_ptr(ggml_init(params));
}

static float get_field(const ggml_tensor * packed, int32_t field, int32_t slot) {
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

static void test_selected_experts_expert_major_order() {
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

    llama_moe_hot_cache_build_worklist(
            packed,
            selected,
            weights,
            layer,
            0,
            1,
            llama_moe_hot_cache_worklist_order::expert_major);

    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID, 0) == 0.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT, 0) == 0.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID, 0) == 0.0f);
    require_close(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT, 0), 0.11f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, 0) == 1.0f);

    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID, 1) == 0.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT, 1) == 3.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID, 1) == 1.0f);
    require_close(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT, 1), 0.22f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, 1) == 1.0f);

    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID, 2) == 1.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT, 2) == 1.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID, 2) == 0.0f);
    require_close(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT, 2), 0.12f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, 2) == 3.0f);

    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID, 0) == 0.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT, 0) == 4.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, 0) == 2.0f);
    require_close(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT, 0), 0.31f);

    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID, 1) == 2.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT, 1) == 2.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, 1) == 1.0f);
    require_close(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT, 1), 0.21f);

    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID, 2) == 2.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT, 2) == 5.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, 2) == 2.0f);
    require_close(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT, 2), 0.32f);

    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_COUNT, 0) == 3.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_COUNT, 0) == 3.0f);
}

static void test_logits_expert_major_order() {
    auto ctx = make_ctx();
    require(ctx != nullptr);

    const int32_t n_expert_used = 2;
    const int32_t n_tokens = 2;
    const int32_t capacity = n_expert_used*n_tokens;

    ggml_tensor * logits = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, n_tokens);
    ggml_tensor * packed = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, capacity, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COUNT);

    set_logit(logits, 0, 0, 0.0f);
    set_logit(logits, 1, 0, 3.0f);
    set_logit(logits, 2, 0, 1.0f);
    set_logit(logits, 3, 0, 4.0f);

    set_logit(logits, 0, 1, 0.0f);
    set_logit(logits, 1, 1, 5.0f);
    set_logit(logits, 2, 1, 4.0f);
    set_logit(logits, 3, 1, 1.0f);

    llama_moe_hot_cache_layer layer;
    layer.n_expert = 4;
    layer.n_hot = 2;
    layer.hot_id_map_host = { -1, -1, 1, 0 };

    llama_moe_hot_cache_build_worklist_from_logits(
            packed,
            logits,
            layer,
            0,
            1,
            llama_moe_hot_cache_worklist_order::expert_major);

    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID, 0) == 0.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT, 0) == 0.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID, 0) == 0.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, 0) == 3.0f);
    require_close(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT, 0), 0.7310586f);

    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID, 1) == 1.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT, 1) == 3.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID, 1) == 1.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, 1) == 2.0f);
    require_close(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT, 1), 0.26894143f);

    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID, 0) == 1.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT, 0) == 1.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, 0) == 0.0f);
    require_close(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT, 0), 0.26894143f);

    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID, 1) == 1.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT, 1) == 2.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, 1) == 1.0f);
    require_close(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT, 1), 0.7310586f);

    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_COUNT, 0) == 2.0f);
    require(get_field(packed, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_COUNT, 0) == 2.0f);
}

int main() {
    test_selected_experts_expert_major_order();
    test_logits_expert_major_order();
    return 0;
}
