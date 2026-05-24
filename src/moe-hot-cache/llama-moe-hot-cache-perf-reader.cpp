#include "llama-moe-hot-cache-perf-reader.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

void llama_moe_layer_perf_tensor_reader::count_topk_locked(
        llama_moe_layer_perf_state & state,
        uint32_t layer,
        ggml_tensor * t) {
    if (t == nullptr) {
        return;
    }

    const int64_t k        = t->ne[0];
    const int64_t n_tokens = t->ne[1];

    if (k <= 0 || n_tokens <= 0) {
        return;
    }

    std::vector<int32_t> ids(k * n_tokens);

    ggml_backend_tensor_get(
        t,
        ids.data(),
        0,
        ids.size() * sizeof(int32_t));

    for (int64_t i = 0; i < k * n_tokens; ++i) {
        const int32_t expert = ids[i];

        if (expert >= 0 && (uint32_t) expert < state.n_expert) {
            state.add_expert_locked(layer, (uint32_t) expert);
        }
    }

    if (layer < state.layers.size()) {
        state.add_locked(state.layers[layer].calls, 1);
    }
}

void llama_moe_layer_perf_tensor_reader::count_worklist_count_locked(
        llama_moe_layer_perf_state & state,
        uint32_t layer,
        ggml_tensor * t,
        bool hot) {
    if (t == nullptr || layer >= state.layers.size()) {
        return;
    }

    if (ggml_nelements(t) <= 0) {
        return;
    }

    uint64_t valid = 0;
    if (t->type == GGML_TYPE_F32) {
        float count = 0.0f;
        ggml_backend_tensor_get(t, &count, 0, sizeof(count));
        valid = count > 0.0f ? (uint64_t) (count + 0.5f) : 0;
    } else if (t->type == GGML_TYPE_I32) {
        int32_t count = 0;
        ggml_backend_tensor_get(t, &count, 0, sizeof(count));
        valid = count > 0 ? (uint64_t) count : 0;
    } else {
        return;
    }

    auto & dst = state.layers[layer];
    if (hot) {
        state.add_locked(dst.hot_worklist_calls, 1);
        if (dst.calls < dst.hot_worklist_calls) {
            state.add_locked(dst.calls, 1);
        }
        state.add_locked(dst.hot_slots_total, valid);
        if (valid == 0) {
            state.add_locked(dst.hot_zero_calls, 1);
        }
    } else {
        state.add_locked(dst.cold_worklist_calls, 1);
        state.add_locked(dst.cold_slots_total, valid);
        if (valid == 0) {
            state.add_locked(dst.cold_zero_calls, 1);
        }
    }
}

void llama_moe_layer_perf_tensor_reader::count_branch_experts_locked(
        llama_moe_layer_perf_state & state,
        uint32_t layer,
        ggml_tensor * t,
        bool hot) {
    if (t == nullptr || layer >= state.layers.size()) {
        return;
    }

    const int64_t n_ids = ggml_nelements(t);
    if (n_ids <= 0) {
        return;
    }

    if (t->type == GGML_TYPE_I32) {
        std::vector<int32_t> ids(n_ids);
        ggml_backend_tensor_get(
            t,
            ids.data(),
            0,
            ids.size() * sizeof(int32_t));

        for (int32_t id : ids) {
            if (id >= 0 && (uint32_t) id < state.n_expert) {
                state.add_branch_expert_locked(layer, (uint32_t) id, hot);
            }
        }
    } else if (t->type == GGML_TYPE_F32) {
        std::vector<float> ids(n_ids);
        ggml_backend_tensor_get(
            t,
            ids.data(),
            0,
            ids.size() * sizeof(float));

        for (float idf : ids) {
            const int32_t id = (int32_t) (idf + 0.5f);
            if (idf >= 0.0f && id >= 0 && (uint32_t) id < state.n_expert) {
                state.add_branch_expert_locked(layer, (uint32_t) id, hot);
            }
        }
    }
}
