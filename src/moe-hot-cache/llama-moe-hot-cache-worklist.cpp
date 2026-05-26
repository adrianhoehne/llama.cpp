#include "llama-moe-hot-cache-worklist.h"

#include "llama-hparams.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

static bool llama_moe_hot_cache_hot_dummy_padding() {
    static const bool enabled = []() {
        const char * env = std::getenv("LLAMA_MOE_HOT_CACHE_HOT_DUMMY_PADDING");
        return env == nullptr || env[0] == '\0' ||
               (std::strcmp(env, "0") != 0 && std::strcmp(env, "off") != 0 && std::strcmp(env, "false") != 0);
    }();

    return enabled;
}

} // namespace

const char * llama_moe_hot_cache_worklist_order_name(llama_moe_hot_cache_worklist_order order) {
    switch (order) {
        case llama_moe_hot_cache_worklist_order::token_major:
            return "token_major";
        case llama_moe_hot_cache_worklist_order::expert_major:
            return "expert_major";
    }

    return "unknown";
}

void llama_moe_hot_cache_build_worklist(
        ggml_tensor * dst,
        const ggml_tensor * selected_experts,
        const ggml_tensor * weights,
        const llama_moe_hot_cache_layer & layer,
        int ith,
        int nth,
        llama_moe_hot_cache_worklist_order order) {
    GGML_UNUSED(nth);

    if (ith != 0) {
        return;
    }

    GGML_ASSERT(dst != nullptr);
    GGML_ASSERT(selected_experts != nullptr);
    GGML_ASSERT(weights != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(selected_experts->type == GGML_TYPE_I32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->nb[0] == sizeof(float));
    GGML_ASSERT(dst->ne[1] == LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COUNT);
    GGML_ASSERT(weights->ne[0] == 1);
    GGML_ASSERT(selected_experts->ne[0] == weights->ne[1]);
    GGML_ASSERT(selected_experts->ne[1] == weights->ne[2]);
    GGML_ASSERT(int64_t(layer.hot_id_map_host.size()) == layer.n_expert);

    const int32_t capacity = dst->ne[0];
    const int32_t n_expert_used = selected_experts->ne[0];
    const int32_t n_tokens = selected_experts->ne[1];
    const int32_t total_slots = n_expert_used * n_tokens;
    const int32_t dummy_src_slot = total_slots;
    const float hot_padding_id =
        llama_moe_hot_cache_hot_dummy_padding() && layer.n_hot > 0 ? float(layer.n_hot) : -1.0f;

    GGML_ASSERT(capacity == total_slots);

    auto set_field = [&](int32_t field, int32_t slot, float value) {
        char * row = (char *) dst->data + field*dst->nb[1];
        *(float *)(row + slot*dst->nb[0]) = value;
    };

    auto fill_field = [&](int32_t field, float value) {
        float * row = (float *) ((char *) dst->data + field*dst->nb[1]);
        std::fill(row, row + capacity, value);
    };

    fill_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID,        hot_padding_id);
    fill_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT,  float(dummy_src_slot));
    fill_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID,  0.0f);
    fill_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT,    0.0f);
    fill_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID,       -1.0f);
    fill_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT, float(dummy_src_slot));
    fill_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, 0.0f);
    fill_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT,   0.0f);
    fill_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, -1.0f);

    int32_t hot_slot = 0;
    int32_t cold_slot = 0;

    const auto selected_expert_at = [&](int32_t iex, int32_t token) {
        return *(const int32_t *) ((const char *) selected_experts->data + iex*selected_experts->nb[0] + token*selected_experts->nb[1]);
    };

    const auto weight_at = [&](int32_t iex, int32_t token) {
        return *(const float *) ((const char *) weights->data + iex*weights->nb[1] + token*weights->nb[2]);
    };

    const auto write_hot = [&](int32_t slot, int32_t token, int32_t iex, int32_t expert, int32_t hot_id) {
        const float weight = weight_at(iex, token);
        const int32_t src_slot = token*n_expert_used + iex;
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID,        slot, float(hot_id));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT,  slot, float(src_slot));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID,  slot, float(token));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT,    slot, weight);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, slot, float(expert));
    };

    const auto write_cold = [&](int32_t slot, int32_t token, int32_t iex, int32_t expert) {
        const float weight = weight_at(iex, token);
        const int32_t src_slot = token*n_expert_used + iex;
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID,       slot, float(expert));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT, slot, float(src_slot));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, slot, float(token));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT,   slot, weight);
    };

    const auto emit_hot = [&](int32_t token, int32_t iex, int32_t expert, int32_t hot_id) {
        write_hot(hot_slot, token, iex, expert, hot_id);
        ++hot_slot;
    };

    const auto emit_cold = [&](int32_t token, int32_t iex, int32_t expert) {
        write_cold(cold_slot, token, iex, expert);
        ++cold_slot;
    };

    if (order == llama_moe_hot_cache_worklist_order::expert_major) {
        GGML_ASSERT(layer.n_hot <= LLAMA_MAX_EXPERTS);
        GGML_ASSERT(layer.n_expert <= LLAMA_MAX_EXPERTS);

        int32_t hot_offsets[LLAMA_MAX_EXPERTS] = {};
        int32_t cold_offsets[LLAMA_MAX_EXPERTS] = {};

        for (int32_t token = 0; token < n_tokens; ++token) {
            for (int32_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert = selected_expert_at(iex, token);
                GGML_ASSERT(expert >= 0);
                GGML_ASSERT(expert < int32_t(layer.hot_id_map_host.size()));

                const int32_t hot_id = layer.hot_id_map_host[expert];
                if (hot_id >= 0) {
                    GGML_ASSERT(hot_id < int32_t(layer.n_hot));
                    ++hot_offsets[hot_id];
                } else {
                    ++cold_offsets[expert];
                }
            }
        }

        int32_t running = 0;
        for (uint32_t ih = 0; ih < layer.n_hot; ++ih) {
            const int32_t start = running;
            running += hot_offsets[ih];
            hot_offsets[ih] = start;
        }
        hot_slot = running;

        running = 0;
        for (uint32_t expert = 0; expert < layer.n_expert; ++expert) {
            const int32_t start = running;
            running += cold_offsets[expert];
            cold_offsets[expert] = start;
        }
        cold_slot = running;

        for (int32_t token = 0; token < n_tokens; ++token) {
            for (int32_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert = selected_expert_at(iex, token);
                const int32_t hot_id = layer.hot_id_map_host[expert];
                if (hot_id >= 0) {
                    const int32_t slot = hot_offsets[hot_id]++;
                    write_hot(slot, token, iex, expert, hot_id);
                } else {
                    const int32_t slot = cold_offsets[expert]++;
                    write_cold(slot, token, iex, expert);
                }
            }
        }
    } else {
        for (int32_t token = 0; token < n_tokens; ++token) {
            for (int32_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert = selected_expert_at(iex, token);
                GGML_ASSERT(expert >= 0);
                GGML_ASSERT(expert < int32_t(layer.hot_id_map_host.size()));

                const int32_t hot_id = layer.hot_id_map_host[expert];
                if (hot_id >= 0) {
                    emit_hot(token, iex, expert, hot_id);
                } else {
                    emit_cold(token, iex, expert);
                }
            }
        }
    }

    GGML_ASSERT(hot_slot  <= capacity);
    GGML_ASSERT(cold_slot <= capacity);

    if (capacity > 0) {
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_COUNT,  0, float(hot_slot));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_COUNT, 0, float(cold_slot));
    }
}

void llama_moe_hot_cache_build_worklist_from_logits(
        ggml_tensor * dst,
        const ggml_tensor * logits,
        const llama_moe_hot_cache_layer & layer,
        int ith,
        int nth,
        llama_moe_hot_cache_worklist_order order) {
    GGML_UNUSED(nth);

    if (ith != 0) {
        return;
    }

    GGML_ASSERT(dst != nullptr);
    GGML_ASSERT(logits != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->nb[0] == sizeof(float));
    GGML_ASSERT(dst->ne[1] == LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COUNT);
    GGML_ASSERT(int64_t(layer.hot_id_map_host.size()) == layer.n_expert);
    GGML_ASSERT(logits->ne[0] == layer.n_expert);

    const int32_t capacity = dst->ne[0];
    const int32_t n_expert = logits->ne[0];
    const int32_t n_tokens = logits->ne[1];
    GGML_ASSERT(n_tokens > 0);
    GGML_ASSERT(capacity % n_tokens == 0);

    const int32_t n_expert_used = capacity / n_tokens;
    const int32_t total_slots = n_expert_used * n_tokens;
    const int32_t dummy_src_slot = total_slots;
    const float hot_padding_id =
        llama_moe_hot_cache_hot_dummy_padding() && layer.n_hot > 0 ? float(layer.n_hot) : -1.0f;
    GGML_ASSERT(capacity == total_slots);
    GGML_ASSERT(n_expert_used > 0);
    GGML_ASSERT(n_expert_used <= n_expert);
    GGML_ASSERT(n_expert_used <= LLAMA_MAX_EXPERTS);

    auto set_field = [&](int32_t field, int32_t slot, float value) {
        char * row = (char *) dst->data + field*dst->nb[1];
        *(float *)(row + slot*dst->nb[0]) = value;
    };

    auto fill_field = [&](int32_t field, float value) {
        float * row = (float *) ((char *) dst->data + field*dst->nb[1]);
        std::fill(row, row + capacity, value);
    };

    fill_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID,        hot_padding_id);
    fill_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT,  float(dummy_src_slot));
    fill_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID,  0.0f);
    fill_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT,    0.0f);
    fill_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID,       -1.0f);
    fill_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT, float(dummy_src_slot));
    fill_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, 0.0f);
    fill_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT,   0.0f);
    fill_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, -1.0f);

    int32_t hot_slot = 0;
    int32_t cold_slot = 0;
    int32_t top_experts[LLAMA_MAX_EXPERTS];
    float top_logits[LLAMA_MAX_EXPERTS];

    const auto write_hot = [&](int32_t slot, int32_t token, int32_t iex, int32_t expert, int32_t hot_id, float weight) {
        const int32_t src_slot = token*n_expert_used + iex;
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID,        slot, float(hot_id));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT,  slot, float(src_slot));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID,  slot, float(token));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT,    slot, weight);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, slot, float(expert));
    };

    const auto write_cold = [&](int32_t slot, int32_t token, int32_t iex, int32_t expert, float weight) {
        const int32_t src_slot = token*n_expert_used + iex;
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID,       slot, float(expert));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT, slot, float(src_slot));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, slot, float(token));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT,   slot, weight);
    };

    const auto emit_hot = [&](int32_t token, int32_t iex, int32_t expert, int32_t hot_id, float weight) {
        write_hot(hot_slot, token, iex, expert, hot_id, weight);
        ++hot_slot;
    };

    const auto emit_cold = [&](int32_t token, int32_t iex, int32_t expert, float weight) {
        write_cold(cold_slot, token, iex, expert, weight);
        ++cold_slot;
    };

    std::vector<int32_t> selected;
    std::vector<float> selected_weights;
    int32_t hot_offsets[LLAMA_MAX_EXPERTS] = {};
    int32_t cold_offsets[LLAMA_MAX_EXPERTS] = {};
    if (order == llama_moe_hot_cache_worklist_order::expert_major) {
        GGML_ASSERT(layer.n_hot <= LLAMA_MAX_EXPERTS);
        GGML_ASSERT(layer.n_expert <= LLAMA_MAX_EXPERTS);
        selected.assign((size_t) capacity, -1);
        selected_weights.assign((size_t) capacity, 0.0f);
    }

    for (int32_t token = 0; token < n_tokens; ++token) {
        for (int32_t i = 0; i < n_expert_used; ++i) {
            top_experts[i] = -1;
            top_logits[i] = -std::numeric_limits<float>::infinity();
        }

        for (int32_t expert = 0; expert < n_expert; ++expert) {
            const float logit = *(const float *) ((const char *) logits->data + expert*logits->nb[0] + token*logits->nb[1]);

            for (int32_t pos = 0; pos < n_expert_used; ++pos) {
                if (logit <= top_logits[pos]) {
                    continue;
                }

                for (int32_t move = n_expert_used - 1; move > pos; --move) {
                    top_logits[move] = top_logits[move - 1];
                    top_experts[move] = top_experts[move - 1];
                }
                top_logits[pos] = logit;
                top_experts[pos] = expert;
                break;
            }
        }

        const float max_logit = top_logits[0];
        float weight_sum = 0.0f;
        for (int32_t iex = 0; iex < n_expert_used; ++iex) {
            top_logits[iex] = std::exp(top_logits[iex] - max_logit);
            weight_sum += top_logits[iex];
        }

        const float weight_scale =
            layer.expert_weights_scale != 0.0f && layer.expert_weights_scale != 1.0f ? layer.expert_weights_scale : 1.0f;

        for (int32_t iex = 0; iex < n_expert_used; ++iex) {
            const int32_t expert = top_experts[iex];
            GGML_ASSERT(expert >= 0);
            GGML_ASSERT(expert < int32_t(layer.hot_id_map_host.size()));

            const float weight = top_logits[iex] / weight_sum * weight_scale;
            const int32_t hot_id = layer.hot_id_map_host[expert];

            if (order == llama_moe_hot_cache_worklist_order::expert_major) {
                const int32_t src_slot = token*n_expert_used + iex;
                selected[(size_t) src_slot] = expert;
                selected_weights[(size_t) src_slot] = weight;
                if (hot_id >= 0) {
                    GGML_ASSERT(hot_id < int32_t(layer.n_hot));
                    ++hot_offsets[hot_id];
                } else {
                    ++cold_offsets[expert];
                }
            } else if (hot_id >= 0) {
                emit_hot(token, iex, expert, hot_id, weight);
            } else {
                emit_cold(token, iex, expert, weight);
            }
        }
    }

    if (order == llama_moe_hot_cache_worklist_order::expert_major) {
        int32_t running = 0;
        for (uint32_t ih = 0; ih < layer.n_hot; ++ih) {
            const int32_t start = running;
            running += hot_offsets[ih];
            hot_offsets[ih] = start;
        }
        hot_slot = running;

        running = 0;
        for (uint32_t expert = 0; expert < layer.n_expert; ++expert) {
            const int32_t start = running;
            running += cold_offsets[expert];
            cold_offsets[expert] = start;
        }
        cold_slot = running;

        for (int32_t token = 0; token < n_tokens; ++token) {
            for (int32_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t src_slot = token*n_expert_used + iex;
                const int32_t expert = selected[(size_t) src_slot];
                GGML_ASSERT(expert >= 0);

                const int32_t hot_id = layer.hot_id_map_host[expert];
                if (hot_id >= 0) {
                    const int32_t slot = hot_offsets[hot_id]++;
                    write_hot(slot, token, iex, expert, hot_id, selected_weights[(size_t) src_slot]);
                } else {
                    const int32_t slot = cold_offsets[expert]++;
                    write_cold(slot, token, iex, expert, selected_weights[(size_t) src_slot]);
                }
            }
        }
    }

    GGML_ASSERT(hot_slot  <= capacity);
    GGML_ASSERT(cold_slot <= capacity);

    if (capacity > 0) {
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_COUNT,  0, float(hot_slot));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_COUNT, 0, float(cold_slot));
    }
}
