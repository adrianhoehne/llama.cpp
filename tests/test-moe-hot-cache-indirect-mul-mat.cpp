#include "ggml.h"
#include "ggml-cpu.h"
#include "../src/moe-hot-cache/llama-moe-hot-cache-ggml.h"

#include <cmath>
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

static void set_f32_3d(ggml_tensor * tensor, int64_t i0, int64_t i1, int64_t i2, float value) {
    *(float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2]) = value;
}

static float get_f32_3d(const ggml_tensor * tensor, int64_t i0, int64_t i1, int64_t i2) {
    return *(const float *) ((const char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2]);
}

static float get_f32_2d(const ggml_tensor * tensor, int64_t i0, int64_t i1) {
    return *(const float *) ((const char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]);
}

static void set_i32_2d(ggml_tensor * tensor, int64_t i0, int64_t i1, int32_t value) {
    *(int32_t *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]) = value;
}

static void set_i32_1d(ggml_tensor * tensor, int64_t i0, int32_t value) {
    *(int32_t *) ((char *) tensor->data + i0*tensor->nb[0]) = value;
}

static void set_allow_negative_ids(ggml_tensor * tensor) {
    const int32_t flags = 1 << 1;
    std::memcpy(tensor->op_params, &flags, sizeof(flags));
}

static void compute_graph(ggml_cgraph * graph) {
    ggml_threadpool_params tpp = ggml_threadpool_params_default(2);
    ggml_threadpool * threadpool = ggml_threadpool_new(&tpp);
    require(threadpool != nullptr);

    ggml_cplan plan = ggml_graph_plan(graph, 2, threadpool);
    std::vector<uint8_t> work_data(plan.work_size);
    plan.work_data = work_data.data();
    ggml_graph_compute(graph, &plan);

    ggml_threadpool_free(threadpool);
}

static void test_indirect_mul_mat_id_matches_materialized_inputs() {
    ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    const int64_t k = 2;
    const int64_t m = 3;
    const int64_t n_experts = 4;
    const int64_t n_tokens = 3;
    const int64_t cold_capacity = 6;

    ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, m, n_experts);
    ggml_tensor * cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, n_tokens);
    ggml_tensor * cold_inputs = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, cold_capacity);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, cold_capacity);
    ggml_tensor * row_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cold_capacity);

    for (int64_t expert = 0; expert < n_experts; ++expert) {
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t col = 0; col < k; ++col) {
                set_f32_3d(weights, col, row, expert, 10.0f*float(expert + 1) + float(row) + 0.25f*float(col));
            }
        }
    }

    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t col = 0; col < k; ++col) {
            set_f32_3d(cur, col, 0, token, 100.0f*float(token + 1) + float(col + 1));
        }
    }

    const int32_t expert_ids[] = { 2, 0, -1, 3, 2, -1 };
    const int32_t token_ids[]  = { 1, 0,  0, 2, 1,  0 };
    for (int64_t slot = 0; slot < cold_capacity; ++slot) {
        set_i32_2d(ids, 0, slot, expert_ids[slot]);
        set_i32_1d(row_ids, slot, token_ids[slot]);

        for (int64_t col = 0; col < k; ++col) {
            set_f32_3d(cold_inputs, col, 0, slot, get_f32_3d(cur, col, 0, token_ids[slot]));
        }
    }

    ggml_tensor * direct = ggml_mul_mat_id(ctx, weights, cold_inputs, ids);
    set_allow_negative_ids(direct);
    ggml_tensor * indirect = llama_moe_hot_cache_ggml_mul_mat_id_indirect(ctx, weights, cur, ids, row_ids);
    set_allow_negative_ids(indirect);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, direct);
    ggml_build_forward_expand(graph, indirect);
    compute_graph(graph);

    for (int64_t slot = 0; slot < cold_capacity; ++slot) {
        for (int64_t row = 0; row < m; ++row) {
            const float a = get_f32_3d(direct, row, 0, slot);
            const float b = get_f32_3d(indirect, row, 0, slot);
            require(std::fabs(a - b) < 1e-5f);
        }
    }

    ggml_free(ctx);
}

static void test_mul_mat_id_token_reduce_matches_manual_reduce() {
    ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    const int64_t k = 2;
    const int64_t m = 3;
    const int64_t n_experts = 4;
    const int64_t n_tokens = 3;
    const int64_t cold_capacity = 6;

    ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, m, n_experts);
    ggml_tensor * activations = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, cold_capacity);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, cold_capacity);
    ggml_tensor * token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cold_capacity);
    ggml_tensor * slot_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, cold_capacity);

    for (int64_t expert = 0; expert < n_experts; ++expert) {
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t col = 0; col < k; ++col) {
                set_f32_3d(weights, col, row, expert, 1.0f + 0.5f*float(expert) + float(row) + 0.25f*float(col));
            }
        }
    }

    const int32_t expert_ids[] = { 2, 0, -1, 3, 2, -1 };
    const int32_t output_tokens[] = { 1, 0, 0, 2, 1, 0 };
    const float router_weights[] = { 0.25f, 0.50f, 0.00f, 1.25f, 0.75f, 0.00f };
    for (int64_t slot = 0; slot < cold_capacity; ++slot) {
        set_i32_2d(ids, 0, slot, expert_ids[slot]);
        set_i32_1d(token_ids, slot, output_tokens[slot]);
        set_f32_3d(slot_weights, 0, 0, slot, router_weights[slot]);

        for (int64_t col = 0; col < k; ++col) {
            set_f32_3d(activations, col, 0, slot, 10.0f*float(slot + 1) + float(col + 1));
        }
    }

    ggml_tensor * direct = ggml_mul_mat_id(ctx, weights, activations, ids);
    set_allow_negative_ids(direct);
    ggml_tensor * fused = llama_moe_hot_cache_ggml_mul_mat_id_token_reduce(
            ctx, weights, activations, ids, token_ids, slot_weights, n_tokens);
    set_allow_negative_ids(fused);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, direct);
    ggml_build_forward_expand(graph, fused);
    compute_graph(graph);

    float expected[3][3] = {};
    for (int64_t slot = 0; slot < cold_capacity; ++slot) {
        if (expert_ids[slot] < 0) {
            continue;
        }

        const int64_t token = output_tokens[slot];
        for (int64_t row = 0; row < m; ++row) {
            expected[token][row] += get_f32_3d(direct, row, 0, slot)*router_weights[slot];
        }
    }

    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t row = 0; row < m; ++row) {
            require(std::fabs(get_f32_2d(fused, row, token) - expected[token][row]) < 1e-5f);
        }
    }

    ggml_free(ctx);
}

int main() {
    test_indirect_mul_mat_id_matches_materialized_inputs();
    test_mul_mat_id_token_reduce_matches_manual_reduce();
    return 0;
}
