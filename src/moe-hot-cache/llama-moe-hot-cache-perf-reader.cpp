#include "llama-moe-hot-cache-perf-reader.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "llama-moe-hot-cache.h"

#include <vector>

namespace {

uint64_t read_f32_count_field(ggml_tensor * t, int32_t field) {
    if (t == nullptr || t->type != GGML_TYPE_F32 || field < 0 || field >= t->ne[1]) {
        return 0;
    }

    float count = 0.0f;
    ggml_backend_tensor_get(t, &count, field*t->nb[1], sizeof(count));
    return count > 0.0f ? (uint64_t) (count + 0.5f) : 0;
}

} // namespace

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

void llama_moe_layer_perf_tensor_reader::count_worklist_counts_locked(
        llama_moe_layer_perf_state & state,
        uint32_t layer,
        ggml_tensor * t,
        bool multi_lane) {
    if (t == nullptr || layer >= state.layers.size() || t->type != GGML_TYPE_F32) {
        return;
    }

    uint64_t hot = read_f32_count_field(t, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_COUNT);
    if (multi_lane) {
        hot += read_f32_count_field(t, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT1_COUNT);
        hot += read_f32_count_field(t, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT2_COUNT);
    }
    const uint64_t cold = read_f32_count_field(t, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_COUNT);

    auto & dst = state.layers[layer];
    state.add_locked(dst.calls, 1);
    state.add_locked(dst.hot_worklist_calls, 1);
    state.add_locked(dst.cold_worklist_calls, 1);
    state.add_locked(dst.hot_slots_total, hot);
    state.add_locked(dst.cold_slots_total, cold);
    if (hot == 0) {
        state.add_locked(dst.hot_zero_calls, 1);
    }
    if (cold == 0) {
        state.add_locked(dst.cold_zero_calls, 1);
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

void llama_moe_layer_perf_tensor_reader::count_hot_lane_worklist_count_locked(
        llama_moe_layer_perf_state & state,
        uint32_t layer,
        ggml_tensor * t,
        uint32_t lane) {
    if (t == nullptr || layer >= state.layers.size() || lane >= LLAMA_MOE_LAYER_PERF_HOT_LANES) {
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
    state.add_locked(dst.hot_lane_worklist_calls[lane], 1);
    state.add_locked(dst.hot_lane_slots_total[lane], valid);
    if (valid == 0) {
        state.add_locked(dst.hot_lane_zero_calls[lane], 1);
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
