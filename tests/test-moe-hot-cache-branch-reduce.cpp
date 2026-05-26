#include "../src/moe-hot-cache/llama-moe-hot-cache-branch-reduce.h"
#include "../src/moe-hot-cache/llama-moe-hot-cache.h"

#include "ggml-cpp.h"

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
        /*.mem_size   =*/ 256*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    return ggml_context_ptr(ggml_init(params));
}

static void set_branch_value(ggml_tensor * tensor, int64_t embd, int64_t slot, float value) {
    *(float *) ((char *) tensor->data + embd*tensor->nb[0] + slot*tensor->nb[1]) = value;
}

static void set_worklist_field(ggml_tensor * worklist, int32_t field, int64_t slot, float value) {
    char * row = (char *) worklist->data + field*worklist->nb[1];
    *(float *) (row + slot*worklist->nb[0]) = value;
}

static float get_output_value(const ggml_tensor * tensor, int64_t embd, int64_t token) {
    return *(const float *) ((const char *) tensor->data + embd*tensor->nb[0] + token*tensor->nb[1]);
}

static void require_close(float actual, float expected) {
    require(std::fabs(actual - expected) < 1e-5f);
}

static void test_compact_cold_reduce_sums_by_token() {
    auto ctx = make_ctx();
    require(ctx != nullptr);

    const int64_t n_embd = 4;
    const int64_t n_tokens = 3;
    const int64_t capacity = 5;

    ggml_tensor * branch_out = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, capacity);
    ggml_tensor * worklist = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, capacity, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COUNT);
    ggml_tensor * output = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, n_tokens);

    for (int64_t slot = 0; slot < capacity; ++slot) {
        for (int64_t embd = 0; embd < n_embd; ++embd) {
            set_branch_value(branch_out, embd, slot, 10.0f*float(slot + 1) + float(embd));
        }
    }

    set_worklist_field(worklist, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_COUNT, 0, 5.0f);
    set_worklist_field(worklist, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, 0, 0.0f);
    set_worklist_field(worklist, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, 1, 2.0f);
    set_worklist_field(worklist, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, 2, 0.0f);
    set_worklist_field(worklist, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, 3, 1.0f);
    set_worklist_field(worklist, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, 4, -1.0f);

    llama_moe_hot_cache_reduce_cold_token_rows(output, branch_out, worklist, 0, 2);
    llama_moe_hot_cache_reduce_cold_token_rows(output, branch_out, worklist, 1, 2);

    for (int64_t embd = 0; embd < n_embd; ++embd) {
        require_close(get_output_value(output, embd, 0), (10.0f + float(embd)) + (30.0f + float(embd)));
        require_close(get_output_value(output, embd, 1), 40.0f + float(embd));
        require_close(get_output_value(output, embd, 2), 20.0f + float(embd));
    }
}

int main() {
    test_compact_cold_reduce_sums_by_token();
    return 0;
}
