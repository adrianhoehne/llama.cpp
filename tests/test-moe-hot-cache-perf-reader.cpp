#include "../src/moe-hot-cache/llama-moe-hot-cache-perf-reader.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
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
        /*.mem_size   =*/ 32*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    return ggml_context_ptr(ggml_init(params));
}

static ggml_backend_ptr make_cpu_backend() {
    ggml_backend_load_all();

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend == nullptr) {
        throw std::runtime_error("CPU backend is not available");
    }

    return ggml_backend_ptr(backend);
}

template<typename T>
static void set_tensor_data(ggml_tensor * tensor, const std::vector<T> & data) {
    ggml_backend_tensor_set(tensor, data.data(), 0, data.size()*sizeof(T));
}

template<typename T>
static void set_tensor_value(ggml_tensor * tensor, T value) {
    ggml_backend_tensor_set(tensor, &value, 0, sizeof(T));
}

static void test_count_topk_counts_valid_ids_and_call() {
    ggml_backend_ptr backend = make_cpu_backend();
    auto ctx = make_ctx();

    ggml_tensor * topk = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 3, 2);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    set_tensor_data<int32_t>(topk, { 1, 3, -1, 2, 4, 0 });

    llama_moe_layer_perf_state state;
    state.ensure_shape_locked(1, 4, 2);

    llama_moe_layer_perf_tensor_reader::count_topk_locked(state, 0, topk);

    require(state.layers[0].calls == 1);
    require(state.layers[0].expert_hits_total == 4);
    require(state.layers[0].experts[0] == 1);
    require(state.layers[0].experts[1] == 1);
    require(state.layers[0].experts[2] == 1);
    require(state.layers[0].experts[3] == 1);

    llama_moe_layer_perf_tensor_reader::count_topk_locked(state, 3, topk);
    require(state.layers[0].calls == 1);
}

static void test_count_worklist_count_reads_f32_and_i32() {
    ggml_backend_ptr backend = make_cpu_backend();
    auto ctx = make_ctx();

    ggml_tensor * hot_count = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_tensor * cold_count = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));

    llama_moe_layer_perf_state state;
    state.ensure_shape_locked(1, 4, 2);

    set_tensor_value<float>(hot_count, 2.6f);
    llama_moe_layer_perf_tensor_reader::count_worklist_count_locked(state, 0, hot_count, true);
    require(state.layers[0].calls == 1);
    require(state.layers[0].hot_worklist_calls == 1);
    require(state.layers[0].hot_slots_total == 3);
    require(state.layers[0].hot_zero_calls == 0);

    set_tensor_value<float>(hot_count, 0.0f);
    llama_moe_layer_perf_tensor_reader::count_worklist_count_locked(state, 0, hot_count, true);
    require(state.layers[0].calls == 2);
    require(state.layers[0].hot_worklist_calls == 2);
    require(state.layers[0].hot_slots_total == 3);
    require(state.layers[0].hot_zero_calls == 1);

    set_tensor_value<int32_t>(cold_count, 5);
    llama_moe_layer_perf_tensor_reader::count_worklist_count_locked(state, 0, cold_count, false);
    require(state.layers[0].calls == 2);
    require(state.layers[0].cold_worklist_calls == 1);
    require(state.layers[0].cold_slots_total == 5);
    require(state.layers[0].cold_zero_calls == 0);

    set_tensor_value<int32_t>(cold_count, -7);
    llama_moe_layer_perf_tensor_reader::count_worklist_count_locked(state, 0, cold_count, false);
    require(state.layers[0].cold_worklist_calls == 2);
    require(state.layers[0].cold_slots_total == 5);
    require(state.layers[0].cold_zero_calls == 1);
}

static void test_count_worklist_count_ignores_invalid_inputs() {
    ggml_backend_ptr backend = make_cpu_backend();
    auto ctx = make_ctx();

    ggml_tensor * unsupported = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I16, 1);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    set_tensor_value<int16_t>(unsupported, 4);

    llama_moe_layer_perf_state state;
    state.ensure_shape_locked(1, 4, 2);

    llama_moe_layer_perf_tensor_reader::count_worklist_count_locked(state, 0, nullptr, true);
    llama_moe_layer_perf_tensor_reader::count_worklist_count_locked(state, 3, unsupported, true);
    llama_moe_layer_perf_tensor_reader::count_worklist_count_locked(state, 0, unsupported, true);

    require(state.layers[0].calls == 0);
    require(state.layers[0].hot_worklist_calls == 0);
    require(state.layers[0].hot_slots_total == 0);
}

static void test_count_branch_experts_reads_i32_and_f32() {
    ggml_backend_ptr backend = make_cpu_backend();
    auto ctx = make_ctx();

    ggml_tensor * hot_ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 5);
    ggml_tensor * cold_ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 5);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));

    set_tensor_data<int32_t>(hot_ids, { -1, 0, 2, 4, 2 });
    set_tensor_data<float>(cold_ids, { -0.2f, 1.49f, 2.60f, 3.60f, 0.0f });

    llama_moe_layer_perf_state state;
    state.ensure_shape_locked(1, 4, 2);

    llama_moe_layer_perf_tensor_reader::count_branch_experts_locked(state, 0, hot_ids, true);
    llama_moe_layer_perf_tensor_reader::count_branch_experts_locked(state, 0, cold_ids, false);

    require(state.layers[0].hot_experts[0] == 1);
    require(state.layers[0].hot_experts[1] == 0);
    require(state.layers[0].hot_experts[2] == 2);
    require(state.layers[0].hot_experts[3] == 0);

    require(state.layers[0].cold_experts[0] == 1);
    require(state.layers[0].cold_experts[1] == 1);
    require(state.layers[0].cold_experts[2] == 0);
    require(state.layers[0].cold_experts[3] == 1);
}

int main() {
    test_count_topk_counts_valid_ids_and_call();
    test_count_worklist_count_reads_f32_and_i32();
    test_count_worklist_count_ignores_invalid_inputs();
    test_count_branch_experts_reads_i32_and_f32();
    return 0;
}
