#include "llama-moe-hot-cache.h"
#include "llama-moe-hot-cache-perf.h"
#include "ggml-backend-moe-hot-cache.h"
#include "models/models.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

enum llama_moe_hot_cache_mul_mat_id_flags : uint32_t {
    LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_NONE                         = 0,
    LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_ALLOW_NEGATIVE_IDS           = 1u << 1,
    // Only valid when rows produced for negative IDs are guaranteed to be ignored later.
    LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_SKIP_NEGATIVE_ID_OUTPUT_ZERO = 1u << 2,
    // Only valid when all selected input rows are duplicates of src1 row 0.
    LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_SHARED_INPUT_ROW             = 1u << 3,
};

struct llama_moe_hot_cache_graph_profile {
    bool cpu_decode_routing = false;
    bool decode_direct_merge = false;
    bool decode_strided_sum_rows = false;
    bool shared_input_row = false;
    bool cold_prefix_sum = false;
    bool cold_prefix_weighted_sum = false;
    bool decode_repeat_hot_input = false;
    bool cold_first_row_input = false;
    bool merge_sum_rows = false;
    bool branch_reduce_merge = false;
};

class llama_moe_hot_cache_graph_tweaks {
public:
    enum pp_reduce_merge_mode {
        PP_REDUCE_MERGE_OFF,
        PP_REDUCE_MERGE_ON,
        PP_REDUCE_MERGE_AUTO,
    };

    static int parallel_mode() {
        const char * env = std::getenv("LLAMA_MOE_HOT_CACHE_PARALLEL");
        if (env == nullptr || env[0] == '\0') {
            return 1;
        }
        if (std::strcmp(env, "0") == 0 || std::strcmp(env, "off") == 0 || std::strcmp(env, "false") == 0) {
            return 0;
        }
        if (std::strcmp(env, "force") == 0) {
            return 2;
        }
        return 1;
    }

    static int64_t parallel_min_slots() {
        static const int64_t value = []() {
            const char * env = std::getenv("LLAMA_MOE_HOT_CACHE_PARALLEL_MIN_SLOTS");
            if (env == nullptr || env[0] == '\0') {
                return int64_t(2);
            }

            char * end = nullptr;
            const long long parsed = std::strtoll(env, &end, 10);
            if (end == env || parsed < 0) {
                return int64_t(2);
            }

            return (int64_t) parsed;
        }();

        return value;
    }

    static bool merge_sum_rows() {
        static const bool enabled = env_enabled_by_default("LLAMA_MOE_HOT_CACHE_MERGE_SUM_ROWS");
        return enabled;
    }

    static bool cpu_decode_routing() {
        static const bool enabled = env_enabled_by_default("LLAMA_MOE_HOT_CACHE_CPU_DECODE_ROUTING");
        return enabled;
    }

    static bool decode_direct_merge() {
        static const bool enabled = env_enabled_by_default("LLAMA_MOE_HOT_CACHE_DECODE_DIRECT_MERGE");
        return enabled;
    }

    static bool decode_strided_sum_rows() {
        static const bool enabled = env_enabled_by_default("LLAMA_MOE_HOT_CACHE_DECODE_STRIDED_SUM_ROWS");
        return enabled;
    }

    static bool hot_dummy_padding() {
        static const bool enabled = env_enabled_by_default("LLAMA_MOE_HOT_CACHE_HOT_DUMMY_PADDING");
        return enabled;
    }

    static bool shared_input_row() {
        static const bool enabled = env_enabled_by_default("LLAMA_MOE_HOT_CACHE_SHARED_INPUT_ROW");
        return enabled;
    }

    static bool cold_prefix_sum() {
        static const bool enabled = env_enabled_by_default("LLAMA_MOE_HOT_CACHE_COLD_PREFIX_SUM");
        return enabled;
    }

    static bool cold_prefix_weighted_sum() {
        static const bool enabled = env_enabled_by_default("LLAMA_MOE_HOT_CACHE_COLD_PREFIX_WEIGHTED_SUM");
        return enabled;
    }

    static bool decode_repeat_hot_input() {
        static const bool enabled = env_enabled_by_default("LLAMA_MOE_HOT_CACHE_DECODE_REPEAT_HOT_INPUT");
        return enabled;
    }

    static bool cold_first_row_input() {
        static const bool enabled = env_enabled_by_default("LLAMA_MOE_HOT_CACHE_COLD_FIRST_ROW_INPUT");
        return enabled;
    }

    static bool branch_reduce_merge() {
        static const bool enabled = env_enabled_by_default("LLAMA_MOE_HOT_CACHE_BRANCH_REDUCE_MERGE");
        return enabled;
    }

    static bool pp_reduce_merge(int64_t n_tokens, int64_t capacity) {
        switch (pp_reduce_merge_mode_value()) {
            case PP_REDUCE_MERGE_OFF:
                return false;
            case PP_REDUCE_MERGE_ON:
                return true;
            case PP_REDUCE_MERGE_AUTO:
                return n_tokens >= 32 && capacity >= 64;
        }

        return false;
    }

private:
    static pp_reduce_merge_mode pp_reduce_merge_mode_value() {
        static const pp_reduce_merge_mode mode = []() {
            const char * env = std::getenv("LLAMA_MOE_HOT_CACHE_PP_REDUCE_MERGE");
            if (env == nullptr || env[0] == '\0' ||
                    std::strcmp(env, "0") == 0 ||
                    std::strcmp(env, "off") == 0 ||
                    std::strcmp(env, "false") == 0) {
                return PP_REDUCE_MERGE_OFF;
            }
            if (std::strcmp(env, "auto") == 0) {
                return PP_REDUCE_MERGE_AUTO;
            }
            return PP_REDUCE_MERGE_ON;
        }();

        return mode;
    }

    static bool env_enabled_by_default(const char * name) {
        const char * env = std::getenv(name);
        return env == nullptr || env[0] == '\0' ||
               (std::strcmp(env, "0") != 0 && std::strcmp(env, "off") != 0 && std::strcmp(env, "false") != 0);
    }
};

class llama_qwen35moe_hot_cache_graph_tweaks {
public:
    static llama_moe_hot_cache_graph_profile get_profile() {
        return {
            /* cpu_decode_routing      = */ llama_moe_hot_cache_graph_tweaks::cpu_decode_routing(),
            /* decode_direct_merge     = */ llama_moe_hot_cache_graph_tweaks::decode_direct_merge(),
            /* decode_strided_sum_rows = */ llama_moe_hot_cache_graph_tweaks::decode_strided_sum_rows(),
            /* shared_input_row        = */ llama_moe_hot_cache_graph_tweaks::shared_input_row(),
            /* cold_prefix_sum         = */ llama_moe_hot_cache_graph_tweaks::cold_prefix_sum(),
            /* cold_prefix_weighted_sum= */ llama_moe_hot_cache_graph_tweaks::cold_prefix_weighted_sum(),
            /* decode_repeat_hot_input = */ llama_moe_hot_cache_graph_tweaks::decode_repeat_hot_input(),
            /* cold_first_row_input    = */ llama_moe_hot_cache_graph_tweaks::cold_first_row_input(),
            /* merge_sum_rows          = */ llama_moe_hot_cache_graph_tweaks::merge_sum_rows(),
            /* branch_reduce_merge     = */ false,
        };
    }
};

class llama_gemma4_hot_cache_graph_tweaks {
public:
    static llama_moe_hot_cache_graph_profile get_profile() {
        // Gemma's cold lane is usually sparse during decode. Use the direct merge
        // path so the CPU lane reduces only the compact cold prefix before joining
        // the hot lane, instead of materializing and reducing the full slot tensor.
        return {
            /* cpu_decode_routing      = */ llama_moe_hot_cache_graph_tweaks::cpu_decode_routing(),
            /* decode_direct_merge     = */ llama_moe_hot_cache_graph_tweaks::decode_direct_merge(),
            /* decode_strided_sum_rows = */ llama_moe_hot_cache_graph_tweaks::decode_strided_sum_rows(),
            /* shared_input_row        = */ llama_moe_hot_cache_graph_tweaks::shared_input_row(),
            /* cold_prefix_sum         = */ llama_moe_hot_cache_graph_tweaks::cold_prefix_sum(),
            /* cold_prefix_weighted_sum= */ llama_moe_hot_cache_graph_tweaks::cold_prefix_weighted_sum(),
            /* decode_repeat_hot_input = */ llama_moe_hot_cache_graph_tweaks::decode_repeat_hot_input(),
            /* cold_first_row_input    = */ llama_moe_hot_cache_graph_tweaks::cold_first_row_input(),
            /* merge_sum_rows          = */ llama_moe_hot_cache_graph_tweaks::merge_sum_rows(),
            /* branch_reduce_merge     = */ llama_moe_hot_cache_graph_tweaks::branch_reduce_merge(),
        };
    }
};

class llama_moe_hot_cache_graph_profiles {
public:
    static llama_moe_hot_cache_graph_profile profile_for_arch(llm_arch arch) {
        switch (arch) {
            case LLM_ARCH_QWEN35MOE:
                return llama_qwen35moe_hot_cache_graph_tweaks::get_profile();
            case LLM_ARCH_GEMMA4:
                return llama_gemma4_hot_cache_graph_tweaks::get_profile();
            default:
                // New MoE architectures must explicitly opt into shortcuts after their graph
                // split order and numerical behavior have been checked.
                return {};
        }
    }
};

static void llama_qwen35moe_hot_cache_build_worklist_op(
        ggml_tensor * dst,
        const ggml_tensor * src0,
        const ggml_tensor * selected_experts,
        const ggml_tensor * weights,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(src0);

    const auto * layer = static_cast<const llama_moe_hot_cache_layer *>(userdata);
    GGML_ASSERT(layer != nullptr);
    llama_moe_hot_cache_build_worklist(dst, selected_experts, weights, *layer, ith, nth);
}

static void llama_qwen35moe_hot_cache_build_worklist_from_logits_op(
        ggml_tensor * dst,
        const ggml_tensor * src0,
        const ggml_tensor * logits,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(src0);

    const auto * layer = static_cast<const llama_moe_hot_cache_layer *>(userdata);
    GGML_ASSERT(layer != nullptr);
    llama_moe_hot_cache_build_worklist_from_logits(dst, logits, *layer, ith, nth);
}

static void llama_qwen35moe_hot_cache_sum_prefix_rows_op(
        ggml_tensor * dst,
        const ggml_tensor * shape,
        const ggml_tensor * src,
        const ggml_tensor * count,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(shape);
    GGML_UNUSED(userdata);

    GGML_ASSERT(dst != nullptr);
    GGML_ASSERT(src != nullptr);
    GGML_ASSERT(count != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(count->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[1] == 1);
    GGML_ASSERT(src->ne[0] == dst->ne[0]);
    GGML_ASSERT(dst->nb[0] == sizeof(float));

    const int64_t n_embd = dst->ne[0];
    const int64_t capacity = src->ne[1];
    const float count_f = *(const float *) count->data;
    int64_t n_prefix = count_f > 0.0f ? (int64_t) (count_f + 0.5f) : 0;
    if (n_prefix > capacity) {
        n_prefix = capacity;
    }

    const int64_t dr = (n_embd + nth - 1)/nth;
    const int64_t i0 = dr*ith;
    const int64_t i1 = std::min(i0 + dr, n_embd);

    for (int64_t i = i0; i < i1; ++i) {
        *(float *) ((char *) dst->data + i*dst->nb[0]) = 0.0f;
    }

    for (int64_t slot = 0; slot < n_prefix; ++slot) {
        for (int64_t i = i0; i < i1; ++i) {
            float * dst_i = (float *) ((char *) dst->data + i*dst->nb[0]);
            const float * src_i = (const float *) ((const char *) src->data + i*src->nb[0] + slot*src->nb[1]);
            *dst_i += *src_i;
        }
    }
}

static void llama_qwen35moe_hot_cache_sum_weighted_prefix_rows_op(
        ggml_tensor * dst,
        const ggml_tensor * shape,
        const ggml_tensor * src,
        const ggml_tensor * worklist,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(shape);
    GGML_UNUSED(userdata);

    GGML_ASSERT(dst != nullptr);
    GGML_ASSERT(src != nullptr);
    GGML_ASSERT(worklist != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(worklist->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[1] == 1);
    GGML_ASSERT(src->ne[0] == dst->ne[0]);
    GGML_ASSERT(worklist->ne[0] >= src->ne[1]);
    GGML_ASSERT(worklist->ne[1] == LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COUNT);
    GGML_ASSERT(dst->nb[0] == sizeof(float));

    const int64_t n_embd = dst->ne[0];
    const int64_t capacity = src->ne[1];
    const char * count_data = (const char *) worklist->data + LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_COUNT*worklist->nb[1];
    const float count_f = *(const float *) count_data;
    int64_t n_prefix = count_f > 0.0f ? (int64_t) (count_f + 0.5f) : 0;
    if (n_prefix > capacity) {
        n_prefix = capacity;
    }

    const int64_t dr = (n_embd + nth - 1)/nth;
    const int64_t i0 = dr*ith;
    const int64_t i1 = std::min(i0 + dr, n_embd);

    for (int64_t i = i0; i < i1; ++i) {
        *(float *) ((char *) dst->data + i*dst->nb[0]) = 0.0f;
    }

    const char * weights_data = (const char *) worklist->data + LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT*worklist->nb[1];
    for (int64_t slot = 0; slot < n_prefix; ++slot) {
        const float weight = *(const float *) (weights_data + slot*worklist->nb[0]);
        for (int64_t i = i0; i < i1; ++i) {
            float * dst_i = (float *) ((char *) dst->data + i*dst->nb[0]);
            const float * src_i = (const float *) ((const char *) src->data + i*src->nb[0] + slot*src->nb[1]);
            *dst_i += *src_i * weight;
        }
    }
}

static void llama_qwen35moe_hot_cache_first_row_input_op(
        ggml_tensor * dst,
        const ggml_tensor * shape,
        const ggml_tensor * src,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(shape);
    GGML_UNUSED(nth);
    GGML_UNUSED(userdata);

    if (ith != 0) {
        return;
    }

    GGML_ASSERT(dst != nullptr);
    GGML_ASSERT(src != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[0] == src->ne[0]);
    GGML_ASSERT(src->ne[1] == 1);
    GGML_ASSERT(dst->nb[0] == sizeof(float));
    GGML_ASSERT(src->nb[0] == sizeof(float));

    memcpy(dst->data, src->data, dst->ne[0]*sizeof(float));
}

static void llama_moe_hot_cache_set_mul_mat_id_flags(ggml_tensor * t, uint32_t flags) {
    if (flags != LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_NONE) {
        memcpy(t->op_params, &flags, sizeof(flags));
    }
}

static ggml_tensor * llama_moe_hot_cache_build_lora_mm_id(
        const llm_graph_context & graph,
        ggml_tensor * w,
        ggml_tensor * cur,
        ggml_tensor * ids,
        uint32_t flags) {
    ggml_tensor * res = ggml_mul_mat_id(graph.ctx0, w, cur, ids);
    llama_moe_hot_cache_set_mul_mat_id_flags(res, flags);

    for (const auto & lora : *graph.loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        ggml_tensor * a_cur = ggml_mul_mat_id(graph.ctx0, lw->a, cur, ids);
        llama_moe_hot_cache_set_mul_mat_id_flags(a_cur, flags);

        ggml_tensor * ab_cur = ggml_mul_mat_id(graph.ctx0, lw->b, a_cur, ids);
        const uint32_t lora_output_flags = flags & ~LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_SHARED_INPUT_ROW;
        llama_moe_hot_cache_set_mul_mat_id_flags(ab_cur, lora_output_flags);

        ab_cur = ggml_scale(graph.ctx0, ab_cur, scale);
        res = ggml_add(graph.ctx0, res, ab_cur);
    }

    return res;
}

} // namespace

static ggml_tensor * llama_moe_hot_cache_build_moe_ffn_with_ids(
        const llm_graph_context & graph,
             ggml_tensor * cur,
             ggml_tensor * selected_experts,
             ggml_tensor * weights,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                 int   il,
             ggml_tensor * gate_up_exps,
             ggml_tensor * up_exps_s,
             ggml_tensor * gate_exps_s,
             ggml_tensor * down_exps_s,
                uint32_t   flags,
             const char * branch_name,
            ggml_backend_t branch_backend = nullptr,
                    bool   apply_weights = true) {
    ggml_context * ctx0 = graph.ctx0;
    ggml_cgraph * gf = graph.gf;
    ggml_backend_sched_t sched = graph.sched;
    const llama_hparams & hparams = graph.hparams;
    const llama_cparams & cparams = graph.cparams;
    const llm_arch arch = graph.arch;

    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4;
    ggml_tensor * selected_experts_scale_ids = selected_experts;
    const auto cb_moe = [&](ggml_tensor * t, const char * name) {
        if (branch_backend != nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, t, branch_backend);
        }

        if (branch_name == nullptr || branch_name[0] == '\0') {
            graph.cb(t, name, il);
            return;
        }

        GGML_ASSERT(std::strncmp(name, "ffn_moe_", 8) == 0);

        char branch_node_name[96];
        const int nwritten = std::snprintf(
                branch_node_name,
                sizeof(branch_node_name),
                "ffn_moe_%s_%s",
                branch_name,
                name + 8);
        GGML_ASSERT(nwritten > 0 && nwritten < int(sizeof(branch_node_name)));
        graph.cb(t, branch_node_name, il);
    };

    if (weight_before_ffn || apply_weights) {
        ggml_build_forward_expand(gf, weights);
    }

    if ((flags & LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_ALLOW_NEGATIVE_IDS) &&
        (up_exps_s != nullptr || gate_exps_s != nullptr || down_exps_s != nullptr)) {
        ggml_tensor * scale_ids_f32 = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);
        scale_ids_f32 = ggml_clamp(ctx0, scale_ids_f32, 0.0f, float(n_expert - 1));
        selected_experts_scale_ids = ggml_cast(ctx0, scale_ids_f32, GGML_TYPE_I32);
        cb_moe(selected_experts_scale_ids, "ffn_moe_scale_ids");
    }

    // Negative-ID rows in gate/up are ignored by the final down projection, so avoid
    // clearing those intermediate outputs. The final down output keeps zeroing unless
    // the caller knows that invalid rows are ignored by the following merge.
    const uint32_t input_flags = weight_before_ffn
        ? (flags & ~LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_SHARED_INPUT_ROW)
        : flags;
    const uint32_t intermediate_flags = (input_flags & LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_ALLOW_NEGATIVE_IDS)
        ? (input_flags | LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_SKIP_NEGATIVE_ID_OUTPUT_ZERO)
        : input_flags;
    const uint32_t output_flags = flags & ~LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_SHARED_INPUT_ROW;

    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

    if (weight_before_ffn) {
        ggml_tensor * repeated = ggml_repeat_4d(ctx0, cur, n_embd, n_expert_used, n_tokens, 1);
        cur = ggml_mul(ctx0, repeated, weights);
        cb_moe(cur, "ffn_moe_weighted");
    }

    ggml_tensor * up = nullptr;
    ggml_tensor * experts = nullptr;

    if (gate_up_exps) {
        ggml_tensor * gate_up = llama_moe_hot_cache_build_lora_mm_id(graph, gate_up_exps, cur, selected_experts, intermediate_flags);
        cb_moe(gate_up, "ffn_moe_gate_up");

        if (up_exps_s) {
            ggml_tensor * s = ggml_reshape_3d(ctx0, up_exps_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, selected_experts_scale_ids);
            gate_up = ggml_mul(ctx0, gate_up, s);
            cb_moe(gate_up, "ffn_moe_gate_up_scaled");
        }

        const int64_t n_ff = gate_up->ne[0] / 2;
        cur = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
        cb_moe(cur, "ffn_moe_gate");
        up  = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
        cb_moe(up, "ffn_moe_up");
    } else {
        up = llama_moe_hot_cache_build_lora_mm_id(graph, up_exps, cur, selected_experts, intermediate_flags);
        cb_moe(up, "ffn_moe_up");

        if (up_exps_s) {
            ggml_tensor * s = ggml_reshape_3d(ctx0, up_exps_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, selected_experts_scale_ids);
            up = ggml_mul(ctx0, up, s);
            cb_moe(up, "ffn_moe_up_scaled");
        }

        if (gate_exps) {
            cur = llama_moe_hot_cache_build_lora_mm_id(graph, gate_exps, cur, selected_experts, intermediate_flags);
            cb_moe(cur, "ffn_moe_gate");
        } else {
            cur = up;
        }

        if (gate_exps_s) {
            ggml_tensor * s = ggml_reshape_3d(ctx0, gate_exps_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, selected_experts_scale_ids);
            cur = ggml_mul(ctx0, cur, s);
            cb_moe(cur, "ffn_moe_gate_scaled");
        }
    }

    const bool has_gate = gate_exps || gate_up_exps;

    switch (type_op) {
        case LLM_FFN_SILU:
            if (has_gate) {
                cur = ggml_swiglu_split(ctx0, cur, up);
                cb_moe(cur, "ffn_moe_swiglu");
            } else {
                cur = ggml_silu(ctx0, cur);
                cb_moe(cur, "ffn_moe_silu");
            } break;
        case LLM_FFN_GELU:
            if (has_gate) {
                cur = ggml_geglu_split(ctx0, cur, up);
                cb_moe(cur, "ffn_moe_geglu");
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb_moe(cur, "ffn_moe_gelu");
            } break;
        case LLM_FFN_RELU:
            if (has_gate) {
                cur = ggml_reglu_split(ctx0, cur, up);
                cb_moe(cur, "ffn_moe_reglu");
            } else {
                cur = ggml_relu(ctx0, cur);
                cb_moe(cur, "ffn_moe_relu");
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    experts = llama_moe_hot_cache_build_lora_mm_id(graph, down_exps, cur, selected_experts, output_flags);
    cb_moe(experts, "ffn_moe_down");

    if (down_exps_s) {
        ggml_tensor * s = ggml_reshape_3d(ctx0, down_exps_s, 1, n_expert, 1);
        s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
        s = ggml_get_rows(ctx0, s, selected_experts_scale_ids);
        experts = ggml_mul(ctx0, experts, s);
        cb_moe(experts, "ffn_moe_down_scaled");
    }

    if (!weight_before_ffn && apply_weights) {
        experts = ggml_mul(ctx0, experts, weights);
        cb_moe(experts, "ffn_moe_weighted");
    }

    ggml_build_forward_expand(gf, experts);

    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };
    assert(n_expert_used > 0);

    const uint32_t n_expert_used_graph = cparams.warmup
        ? std::min<uint32_t>(hparams.n_expert_used, uint32_t(n_expert_used))
        : uint32_t(n_expert_used);
    GGML_ASSERT(n_expert_used_graph > 0);
    GGML_ASSERT(n_expert_used_graph <= uint32_t(n_expert_used));

    // Warmup may request the baseline top-k view count, but compact MoE paths can
    // legitimately fold the expert-slot axis down to 1, so clamp to the tensor shape.
    for (uint32_t i = 0; i < n_expert_used_graph; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);
        ggml_build_forward_expand(gf, cur_experts[i]);
    }

    ggml_tensor * moe_out = cur_experts[0];
    for (uint32_t i = 1; i < n_expert_used_graph; ++i) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
        ggml_build_forward_expand(gf, moe_out);
    }

    if (n_expert_used_graph == 1) {
        moe_out = ggml_cont(ctx0, moe_out);
    }

    cb_moe(moe_out, "ffn_moe_out");
    return moe_out;
}

static ggml_tensor * llama_moe_hot_cache_build_moe_hot_from_logits(
        const llm_graph_context & graph,
        const llama_model & model,
             ggml_tensor * cur,
             ggml_tensor * logits,
                 int   il,
     llm_ffn_op_type   type_op) {
    ggml_context * ctx0 = graph.ctx0;
    ggml_cgraph * gf = graph.gf;
    ggml_backend_sched_t sched = graph.sched;
    ggml_backend_t backend_cpu = graph.backend_cpu;
    const llama_hparams & hparams = graph.hparams;
    const llama_cparams & cparams = graph.cparams;
    const int64_t n_embd = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const int64_t n_expert = graph.n_expert;
    const int64_t n_expert_used = graph.n_expert_used;
    const int64_t n_moe_slots = cparams.warmup ? hparams.n_expert_used : n_expert_used;
    const auto & layer = model.layers[il];
    const auto & cache = model.moe_hot_cache->layers[il];
    const llama_moe_hot_cache_graph_profile profile = llama_moe_hot_cache_graph_profiles::profile_for_arch(graph.arch);

    GGML_ASSERT(!cache.hot_id_map_host.empty());
    GGML_ASSERT(n_moe_slots > 0);
    GGML_ASSERT(n_moe_slots <= LLAMA_MAX_EXPERTS);

    const int64_t capacity = n_moe_slots*n_tokens;

    ggml_tensor * worklist_shape = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, capacity, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COUNT);
    ggml_tensor * worklist = nullptr;
    if (!cparams.warmup && n_tokens == 1 && profile.cpu_decode_routing) {
        worklist = ggml_map_custom2(
                ctx0,
                worklist_shape,
                logits,
                llama_qwen35moe_hot_cache_build_worklist_from_logits_op,
                1,
                const_cast<llama_moe_hot_cache_layer *>(&cache));
    } else {
        ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, logits, n_moe_slots);
        graph.cb(selected_experts->src[0], "ffn_moe_argsort", il);
        graph.cb(selected_experts, "ffn_moe_topk", il);

        ggml_tensor * logits_rows = ggml_reshape_3d(ctx0, logits, 1, n_expert, n_tokens);
        ggml_tensor * weights = ggml_get_rows(ctx0, logits_rows, selected_experts);
        graph.cb(weights, "ffn_moe_weights", il);

        weights = ggml_reshape_2d(ctx0, weights, n_moe_slots, n_tokens);
        weights = ggml_soft_max(ctx0, weights);
        graph.cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_moe_slots, n_tokens);

        if (hparams.expert_weights_scale != 0.0f && hparams.expert_weights_scale != 1.0f) {
            weights = ggml_scale(ctx0, weights, hparams.expert_weights_scale);
            graph.cb(weights, "ffn_moe_weights_scaled", il);
        }

        worklist = ggml_map_custom3(
                ctx0,
                worklist_shape,
                selected_experts,
                weights,
                llama_qwen35moe_hot_cache_build_worklist_op,
                1,
                const_cast<llama_moe_hot_cache_layer *>(&cache));
    }
    graph.cb(worklist, "ffn_moe_worklist", il);

    if (capacity == 0) {
        graph.cb(cur, "ffn_moe_out", il);
        return cur;
    }

    const auto view_worklist_field = [&](int32_t field) {
        return ggml_view_1d(ctx0, worklist, capacity, field*worklist->nb[1]);
    };

    const int parallel_mode = llama_moe_hot_cache_graph_tweaks::parallel_mode();
    const int64_t parallel_min_slots = llama_moe_hot_cache_graph_tweaks::parallel_min_slots();
    const bool annotate_parallel_region =
        parallel_mode == 2 || parallel_min_slots == 0 || capacity >= parallel_min_slots;
    const bool decode_direct_merge =
        !cparams.warmup && n_tokens == 1 && profile.decode_direct_merge;
    const bool cold_prefix_merge =
        cache.n_hot != cache.n_expert && decode_direct_merge && profile.cold_prefix_sum;
    const bool cold_prefix_weighted_sum =
        cold_prefix_merge && profile.cold_prefix_weighted_sum;
    const bool cold_shared_input_row =
        cache.n_hot != cache.n_expert && !cparams.warmup && n_tokens == 1 && profile.shared_input_row;
    const bool cold_first_row_input =
        cold_shared_input_row && profile.cold_first_row_input;
    const bool perf_expert_counts = llama_moe_layer_perf_needs_expert_counts(cparams.no_perf);
    const bool perf_or_parallel_counts =
        !cparams.warmup && (perf_expert_counts || (parallel_mode != 0 && annotate_parallel_region));
    const bool repeat_hot_input =
        !cparams.warmup && n_tokens == 1 && profile.decode_repeat_hot_input;
    const bool branch_reduce_merge =
        !cparams.warmup && cache.n_hot != cache.n_expert && !decode_direct_merge && n_moe_slots > 1 &&
        ((n_tokens == 1 && profile.branch_reduce_merge) ||
         (n_tokens > 1 && llama_moe_hot_cache_graph_tweaks::pp_reduce_merge(n_tokens, capacity)));

    ggml_tensor * hot_count = nullptr;
    ggml_tensor * cold_count = nullptr;

    if (perf_or_parallel_counts) {
        const auto view_worklist_count = [&](int32_t field) {
            return ggml_view_1d(ctx0, worklist, 1, field*worklist->nb[1]);
        };

        hot_count = view_worklist_count(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_COUNT);
        graph.cb(hot_count, "ffn_moe_hot_count", il);

        cold_count = view_worklist_count(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_COUNT);
        graph.cb(cold_count, "ffn_moe_cold_count", il);
    }

    ggml_tensor * hot_ids = ggml_cast(ctx0, view_worklist_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID), GGML_TYPE_I32);
    hot_ids = ggml_reshape_2d(ctx0, hot_ids, 1, capacity);
    graph.cb(hot_ids, "ffn_moe_hot_ids_compact", il);

    ggml_tensor * hot_src_slots = nullptr;
    if (!decode_direct_merge) {
        hot_src_slots = ggml_cast(ctx0, view_worklist_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT), GGML_TYPE_I32);
        graph.cb(hot_src_slots, "ffn_moe_hot_src_slots", il);
    }

    ggml_tensor * hot_token_ids = nullptr;
    if (!repeat_hot_input) {
        hot_token_ids = ggml_cast(ctx0, view_worklist_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID), GGML_TYPE_I32);
        graph.cb(hot_token_ids, "ffn_moe_hot_token_ids", il);
    }

    ggml_tensor * hot_weights = ggml_reshape_3d(ctx0, view_worklist_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT), 1, 1, capacity);
    graph.cb(hot_weights, "ffn_moe_hot_weights_compact", il);

    ggml_tensor * hot_expert_ids = nullptr;
    if (perf_expert_counts) {
        hot_expert_ids = view_worklist_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID);
        graph.cb(hot_expert_ids, "ffn_moe_hot_expert_ids_compact", il);
    }

    ggml_tensor * cold_ids = nullptr;
    ggml_tensor * cold_src_slots = nullptr;
    ggml_tensor * cold_token_ids = nullptr;
    ggml_tensor * cold_weights = nullptr;

    if (cache.n_hot != cache.n_expert) {
        cold_ids = ggml_cast(ctx0, view_worklist_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID), GGML_TYPE_I32);
        cold_ids = ggml_reshape_2d(ctx0, cold_ids, 1, capacity);
        graph.cb(cold_ids, "ffn_moe_cold_ids_compact", il);

        if (!decode_direct_merge) {
            cold_src_slots = ggml_cast(ctx0, view_worklist_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT), GGML_TYPE_I32);
            graph.cb(cold_src_slots, "ffn_moe_cold_src_slots", il);
        }

        if (!cold_first_row_input) {
            cold_token_ids = ggml_cast(ctx0, view_worklist_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID), GGML_TYPE_I32);
            graph.cb(cold_token_ids, "ffn_moe_cold_token_ids", il);
        }

        if (!cold_prefix_weighted_sum) {
            cold_weights = ggml_reshape_3d(ctx0, view_worklist_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT), 1, 1, capacity);
            graph.cb(cold_weights, "ffn_moe_cold_weights_compact", il);
        }
    }

    if (hot_count != nullptr) {
        ggml_build_forward_expand(gf, hot_count);
    }
    if (cold_count != nullptr) {
        ggml_build_forward_expand(gf, cold_count);
    }
    ggml_build_forward_expand(gf, hot_ids);
    if (hot_src_slots != nullptr) {
        ggml_build_forward_expand(gf, hot_src_slots);
    }
    if (hot_token_ids != nullptr) {
        ggml_build_forward_expand(gf, hot_token_ids);
    }
    ggml_build_forward_expand(gf, hot_weights);
    if (hot_expert_ids != nullptr) {
        ggml_build_forward_expand(gf, hot_expert_ids);
    }
    if (cold_ids != nullptr) {
        ggml_build_forward_expand(gf, cold_ids);
        if (cold_src_slots != nullptr) {
            ggml_build_forward_expand(gf, cold_src_slots);
        }
        if (cold_token_ids != nullptr) {
            ggml_build_forward_expand(gf, cold_token_ids);
        }
        if (cold_weights != nullptr) {
            ggml_build_forward_expand(gf, cold_weights);
        }
    }

    ggml_tensor * hot_inputs = repeat_hot_input
        ? ggml_repeat_4d(ctx0, cur, n_embd, capacity, 1, 1)
        : ggml_get_rows(ctx0, cur, hot_token_ids);
    graph.cb(hot_inputs, "ffn_moe_hot_inputs", il);

    const uint32_t hot_mul_mat_id_flags = llama_moe_hot_cache_graph_tweaks::hot_dummy_padding()
        ? LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_NONE
        : LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_ALLOW_NEGATIVE_IDS;

    const auto merge_compact_slots = [&](
            ggml_tensor * branch_out,
            ggml_backend_t branch_backend,
            ggml_tensor * prefix_count,
            ggml_tensor * prefix_worklist,
            bool prefix_weighted,
            const char * name) {
        if (prefix_count != nullptr && branch_backend != nullptr && n_tokens == 1 &&
                profile.cold_prefix_sum) {
            ggml_tensor * merged_shape = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
            ggml_backend_sched_set_tensor_backend(sched, merged_shape, branch_backend);

            const int prefix_reduce_tasks = std::max<int>(1, std::min<int>(4, cparams.n_threads));
            ggml_tensor * merged = ggml_map_custom3(
                    ctx0,
                    merged_shape,
                    branch_out,
                    prefix_weighted ? prefix_worklist : prefix_count,
                    prefix_weighted
                        ? llama_qwen35moe_hot_cache_sum_weighted_prefix_rows_op
                        : llama_qwen35moe_hot_cache_sum_prefix_rows_op,
                    prefix_reduce_tasks,
                    nullptr);
            ggml_backend_sched_set_tensor_backend(sched, merged, branch_backend);
            graph.cb(merged, name, il);
            ggml_build_forward_expand(gf, merged);
            return merged;
        }

        ggml_tensor * merged = ggml_reshape_3d(ctx0, branch_out, n_embd, capacity, 1);
        if (branch_backend != nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, merged, branch_backend);
        }
        merged = ggml_permute(ctx0, merged, 1, 0, 2, 3);
        if (branch_backend != nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, merged, branch_backend);
        }
        if (!profile.decode_strided_sum_rows) {
            merged = ggml_cont(ctx0, merged);
            if (branch_backend != nullptr) {
                ggml_backend_sched_set_tensor_backend(sched, merged, branch_backend);
            }
        }
        merged = ggml_sum_rows(ctx0, merged);
        if (branch_backend != nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, merged, branch_backend);
        }
        merged = ggml_reshape_2d(ctx0, merged, n_embd, n_tokens);
        if (branch_backend != nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, merged, branch_backend);
        }
        graph.cb(merged, name, il);
        ggml_build_forward_expand(gf, merged);
        return merged;
    };

    const auto reduce_slots_in_branch = [&](
            ggml_tensor * slots,
            ggml_backend_t branch_backend,
            const char * name) {
        ggml_tensor * reduced = ggml_permute(ctx0, slots, 1, 0, 2, 3);
        if (branch_backend != nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, reduced, branch_backend);
        }
        if (!profile.decode_strided_sum_rows) {
            reduced = ggml_cont(ctx0, reduced);
            if (branch_backend != nullptr) {
                ggml_backend_sched_set_tensor_backend(sched, reduced, branch_backend);
            }
        }
        reduced = ggml_sum_rows(ctx0, reduced);
        if (branch_backend != nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, reduced, branch_backend);
        }
        reduced = ggml_reshape_2d(ctx0, reduced, n_embd, n_tokens);
        if (branch_backend != nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, reduced, branch_backend);
        }
        graph.cb(reduced, name, il);
        ggml_build_forward_expand(gf, reduced);
        return reduced;
    };

    ggml_tensor * hot_out = llama_moe_hot_cache_build_moe_ffn_with_ids(graph,
            hot_inputs,
            hot_ids,
            hot_weights,
            cache.ffn_up_exps,
            cache.ffn_gate_exps,
            cache.ffn_down_exps,
            cache.n_hot + 1,
            1,
            type_op,
            il,
            cache.ffn_gate_up_exps,
            cache.ffn_up_exps_s,
            cache.ffn_gate_exps_s,
            cache.ffn_down_exps_s,
            hot_mul_mat_id_flags,
            "hot");
    graph.cb(hot_out, "ffn_moe_hot_out", il);

    ggml_tensor * hot_slots = nullptr;
    if (decode_direct_merge) {
        hot_slots = merge_compact_slots(hot_out, nullptr, nullptr, nullptr, false, "ffn_moe_hot_slots");
    } else {
        hot_slots = ggml_scale(ctx0, hot_inputs, 0.0f);
        hot_slots = ggml_pad(ctx0, hot_slots, 0, 1, 0, 0);
        hot_slots = ggml_set_rows(ctx0, hot_slots, hot_out, hot_src_slots);
        hot_slots = ggml_view_2d(ctx0, hot_slots, n_embd, capacity, hot_slots->nb[1], 0);
        hot_slots = ggml_reshape_3d(ctx0, hot_slots, n_embd, n_moe_slots, n_tokens);
        graph.cb(hot_slots, "ffn_moe_hot_slots", il);
        ggml_build_forward_expand(gf, hot_slots);
    }
    ggml_tensor * hot_parallel_output = hot_slots;
    if (branch_reduce_merge) {
        hot_parallel_output = reduce_slots_in_branch(hot_slots, nullptr, "ffn_moe_hot_slots_reduced");
    }

    ggml_tensor * out_slots = hot_slots;

    if (cache.n_hot != cache.n_expert) {
        ggml_backend_t cold_branch_backend = cparams.warmup ? nullptr : backend_cpu;

        ggml_tensor * cold_inputs = nullptr;
        if (cold_first_row_input) {
            ggml_tensor * cold_inputs_shape = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, capacity);
            ggml_backend_sched_set_tensor_backend(sched, cold_inputs_shape, cold_branch_backend);
            cold_inputs = ggml_map_custom2(
                    ctx0,
                    cold_inputs_shape,
                    cur,
                    llama_qwen35moe_hot_cache_first_row_input_op,
                    1,
                    nullptr);
        } else {
            cold_inputs = ggml_get_rows(ctx0, cur, cold_token_ids);
        }
        if (cold_shared_input_row && !cold_first_row_input) {
            const int32_t get_rows_flags = GGML_GET_ROWS_FLAG_FIRST_ROW_ONLY;
            memcpy(cold_inputs->op_params, &get_rows_flags, sizeof(get_rows_flags));
        }
        graph.cb(cold_inputs, "ffn_moe_cold_inputs", il);
        if (cold_branch_backend != nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, cold_inputs, cold_branch_backend);
        }

        const uint32_t cold_mul_mat_id_flags = LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_ALLOW_NEGATIVE_IDS |
            (cold_shared_input_row ? LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_SHARED_INPUT_ROW : LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_NONE) |
            (cold_prefix_merge ? LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_SKIP_NEGATIVE_ID_OUTPUT_ZERO : LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_NONE);

        ggml_tensor * cold_out = llama_moe_hot_cache_build_moe_ffn_with_ids(graph,
                cold_inputs,
                cold_ids,
                cold_weights,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                n_expert,
                1,
                type_op,
                il,
                layer.ffn_gate_up_exps,
                layer.ffn_up_exps_s,
                layer.ffn_gate_exps_s,
                layer.ffn_down_exps_s,
                cold_mul_mat_id_flags,
                "cold",
                cold_branch_backend,
                !cold_prefix_weighted_sum);
        graph.cb(cold_out, "ffn_moe_cold_out", il);

        ggml_tensor * cold_slots = nullptr;
        if (decode_direct_merge) {
            cold_slots = merge_compact_slots(cold_out, cold_branch_backend, cold_count, worklist, cold_prefix_weighted_sum, "ffn_moe_cold_slots");
        } else {
            ggml_tensor * cold_slots_zero_src = cold_shared_input_row ? cold_out : cold_inputs;
            cold_slots = ggml_scale(ctx0, cold_slots_zero_src, 0.0f);
            if (cold_branch_backend != nullptr) {
                ggml_backend_sched_set_tensor_backend(sched, cold_slots, cold_branch_backend);
            }
            cold_slots = ggml_pad(ctx0, cold_slots, 0, 1, 0, 0);
            if (cold_branch_backend != nullptr) {
                ggml_backend_sched_set_tensor_backend(sched, cold_slots, cold_branch_backend);
            }
            cold_slots = ggml_set_rows(ctx0, cold_slots, cold_out, cold_src_slots);
            if (cold_branch_backend != nullptr) {
                ggml_backend_sched_set_tensor_backend(sched, cold_slots, cold_branch_backend);
            }
            cold_slots = ggml_view_2d(ctx0, cold_slots, n_embd, capacity, cold_slots->nb[1], 0);
            if (cold_branch_backend != nullptr) {
                ggml_backend_sched_set_tensor_backend(sched, cold_slots, cold_branch_backend);
            }
            cold_slots = ggml_reshape_3d(ctx0, cold_slots, n_embd, n_moe_slots, n_tokens);
            graph.cb(cold_slots, "ffn_moe_cold_slots", il);
            if (cold_branch_backend != nullptr) {
                ggml_backend_sched_set_tensor_backend(sched, cold_slots, cold_branch_backend);
            }
            ggml_build_forward_expand(gf, cold_slots);
        }

        ggml_tensor * cold_parallel_output = cold_slots;
        if (branch_reduce_merge) {
            cold_parallel_output = reduce_slots_in_branch(cold_slots, cold_branch_backend, "ffn_moe_cold_slots_reduced");
        }

        out_slots = branch_reduce_merge
            ? ggml_add(ctx0, hot_parallel_output, cold_parallel_output)
            : ggml_add(ctx0, hot_slots, cold_slots);
        graph.cb(out_slots, "ffn_moe_hot_cold_slots", il);

        if (parallel_mode != 0 && !cparams.warmup && annotate_parallel_region && hot_count != nullptr && cold_count != nullptr) {
            const uint32_t parallel_flags = (branch_reduce_merge || cold_prefix_merge)
                ? GGML_BACKEND_SCHED_MOE_HOT_CACHE_PARALLEL_FLAG_ALLOW_JOIN_BRIDGE
                : GGML_BACKEND_SCHED_MOE_HOT_CACHE_PARALLEL_FLAG_NONE;
            ggml_backend_sched_moe_hot_cache_parallel_region(
                    sched,
                    il,
                    parallel_mode,
                    capacity,
                    hot_count,
                    cold_count,
                    hot_inputs,
                    hot_parallel_output,
                    hot_parallel_output,
                    cold_inputs,
                    cold_parallel_output,
                    cold_parallel_output,
                    out_slots,
                    parallel_flags);
        }
    }

    ggml_tensor * out = nullptr;
    if (decode_direct_merge) {
        out = out_slots;
        graph.cb(out, "ffn_moe_out", il);
        return out;
    }

    if (branch_reduce_merge) {
        out = out_slots;
        graph.cb(out, "ffn_moe_out", il);
        return out;
    }

    if (n_moe_slots > 1 && profile.merge_sum_rows) {
        out = ggml_permute(ctx0, out_slots, 1, 0, 2, 3);
        if (!profile.decode_strided_sum_rows) {
            out = ggml_cont(ctx0, out);
        }
        out = ggml_sum_rows(ctx0, out);
        out = ggml_reshape_2d(ctx0, out, n_embd, n_tokens);
        ggml_build_forward_expand(gf, out);
    } else {
        ggml_tensor * slot_outputs[LLAMA_MAX_EXPERTS] = { nullptr };
        for (int64_t i = 0; i < n_moe_slots; ++i) {
            slot_outputs[i] = ggml_view_2d(ctx0, out_slots, n_embd, n_tokens, out_slots->nb[2], i*out_slots->nb[1]);
            ggml_build_forward_expand(gf, slot_outputs[i]);
        }

        out = slot_outputs[0];
        for (int64_t i = 1; i < n_moe_slots; ++i) {
            out = ggml_add(ctx0, out, slot_outputs[i]);
            ggml_build_forward_expand(gf, out);
        }

        if (n_moe_slots == 1) {
            out = ggml_cont(ctx0, out);
        }
    }

    graph.cb(out, "ffn_moe_out", il);
    return out;
}

ggml_tensor * llama_model_qwen35moe::graph::build_layer_ffn_hot(ggml_tensor * cur, const int il) {
    const int64_t n_embd = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const int64_t n_moe_slots = cparams.warmup ? hparams.n_expert_used : n_expert_used;
    const auto & layer = model.layers[il];
    const auto & cache = model.moe_hot_cache->layers[il];

    GGML_ASSERT(!cache.hot_id_map_host.empty());
    GGML_ASSERT(n_moe_slots > 0);
    GGML_ASSERT(n_moe_slots <= LLAMA_MAX_EXPERTS);

    ggml_tensor * logits = build_lora_mm(layer.ffn_gate_inp, cur);
    cb(logits, "ffn_moe_logits", il);

    const int64_t capacity = n_moe_slots*n_tokens;

    ggml_tensor * worklist_shape = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, capacity, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COUNT);
    ggml_tensor * worklist = nullptr;
    if (!cparams.warmup && n_tokens == 1 && llama_moe_hot_cache_graph_tweaks::cpu_decode_routing()) {
        worklist = ggml_map_custom2(
                ctx0,
                worklist_shape,
                logits,
                llama_qwen35moe_hot_cache_build_worklist_from_logits_op,
                1,
                const_cast<llama_moe_hot_cache_layer *>(&cache));
    } else {
        ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, logits, n_moe_slots);
        cb(selected_experts->src[0], "ffn_moe_argsort", il);
        cb(selected_experts, "ffn_moe_topk", il);

        ggml_tensor * logits_rows = ggml_reshape_3d(ctx0, logits, 1, n_expert, n_tokens);
        ggml_tensor * weights = ggml_get_rows(ctx0, logits_rows, selected_experts);
        cb(weights, "ffn_moe_weights", il);

        weights = ggml_reshape_2d(ctx0, weights, n_moe_slots, n_tokens);
        // Top-k after softmax is equivalent to top-k on logits. After the selection,
        // the original full-softmax denominator cancels during top-k renormalization.
        weights = ggml_soft_max(ctx0, weights);
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_moe_slots, n_tokens);

        if (hparams.expert_weights_scale != 0.0f && hparams.expert_weights_scale != 1.0f) {
            weights = ggml_scale(ctx0, weights, hparams.expert_weights_scale);
            cb(weights, "ffn_moe_weights_scaled", il);
        }

        worklist = ggml_map_custom3(
                ctx0,
                worklist_shape,
                selected_experts,
                weights,
                llama_qwen35moe_hot_cache_build_worklist_op,
                1,
                const_cast<llama_moe_hot_cache_layer *>(&cache));
    }
    cb(worklist, "ffn_moe_worklist", il);

    if (capacity == 0) {
        cb(cur, "ffn_moe_out", il);
        return cur;
    }

    const auto view_worklist_field = [&](int32_t field) {
        return ggml_view_1d(ctx0, worklist, capacity, field*worklist->nb[1]);
    };

    const int parallel_mode = llama_moe_hot_cache_graph_tweaks::parallel_mode();
    const int64_t parallel_min_slots = llama_moe_hot_cache_graph_tweaks::parallel_min_slots();
    const bool annotate_parallel_region =
        parallel_mode == 2 || parallel_min_slots == 0 || capacity >= parallel_min_slots;
    const bool decode_direct_merge =
        !cparams.warmup && n_tokens == 1 && llama_moe_hot_cache_graph_tweaks::decode_direct_merge();
    const bool cold_prefix_merge =
        cache.n_hot != cache.n_expert && decode_direct_merge && llama_moe_hot_cache_graph_tweaks::cold_prefix_sum();
    const bool cold_prefix_weighted_sum =
        cold_prefix_merge && llama_moe_hot_cache_graph_tweaks::cold_prefix_weighted_sum();
    const bool cold_shared_input_row =
        cache.n_hot != cache.n_expert && decode_direct_merge && llama_moe_hot_cache_graph_tweaks::shared_input_row();
    const bool cold_first_row_input =
        cold_shared_input_row && llama_moe_hot_cache_graph_tweaks::cold_first_row_input();
    const bool perf_expert_counts = llama_moe_layer_perf_needs_expert_counts(cparams.no_perf);
    const bool perf_or_parallel_counts =
        !cparams.warmup && (perf_expert_counts || (parallel_mode != 0 && annotate_parallel_region));
    const bool repeat_hot_input =
        decode_direct_merge && llama_moe_hot_cache_graph_tweaks::decode_repeat_hot_input();
    const bool branch_reduce_merge =
        !cparams.warmup && n_tokens > 1 && cache.n_hot != cache.n_expert && n_moe_slots > 1 &&
        llama_moe_hot_cache_graph_tweaks::pp_reduce_merge(n_tokens, capacity);

    ggml_tensor * hot_count = nullptr;
    ggml_tensor * cold_count = nullptr;

    if (perf_or_parallel_counts) {
        const auto view_worklist_count = [&](int32_t field) {
            return ggml_view_1d(ctx0, worklist, 1, field*worklist->nb[1]);
        };

        hot_count = view_worklist_count(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_COUNT);
        cb(hot_count, "ffn_moe_hot_count", il);

        cold_count = view_worklist_count(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_COUNT);
        cb(cold_count, "ffn_moe_cold_count", il);
    }

    ggml_tensor * hot_ids = ggml_cast(ctx0, view_worklist_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID), GGML_TYPE_I32);
    hot_ids = ggml_reshape_2d(ctx0, hot_ids, 1, capacity);
    cb(hot_ids, "ffn_moe_hot_ids_compact", il);

    ggml_tensor * hot_src_slots = nullptr;
    if (!decode_direct_merge) {
        hot_src_slots = ggml_cast(ctx0, view_worklist_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT), GGML_TYPE_I32);
        cb(hot_src_slots, "ffn_moe_hot_src_slots", il);
    }

    ggml_tensor * hot_token_ids = nullptr;
    if (!repeat_hot_input) {
        hot_token_ids = ggml_cast(ctx0, view_worklist_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID), GGML_TYPE_I32);
        cb(hot_token_ids, "ffn_moe_hot_token_ids", il);
    }

    ggml_tensor * hot_weights = ggml_reshape_3d(ctx0, view_worklist_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT), 1, 1, capacity);
    cb(hot_weights, "ffn_moe_hot_weights_compact", il);

    ggml_tensor * hot_expert_ids = nullptr;
    if (perf_expert_counts) {
        hot_expert_ids = view_worklist_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID);
        cb(hot_expert_ids, "ffn_moe_hot_expert_ids_compact", il);
    }

    ggml_tensor * cold_ids = nullptr;
    ggml_tensor * cold_src_slots = nullptr;
    ggml_tensor * cold_token_ids = nullptr;
    ggml_tensor * cold_weights = nullptr;

    if (cache.n_hot != cache.n_expert) {
        cold_ids = ggml_cast(ctx0, view_worklist_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID), GGML_TYPE_I32);
        cold_ids = ggml_reshape_2d(ctx0, cold_ids, 1, capacity);
        cb(cold_ids, "ffn_moe_cold_ids_compact", il);

        if (!decode_direct_merge) {
            cold_src_slots = ggml_cast(ctx0, view_worklist_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT), GGML_TYPE_I32);
            cb(cold_src_slots, "ffn_moe_cold_src_slots", il);
        }

        if (!cold_first_row_input) {
            cold_token_ids = ggml_cast(ctx0, view_worklist_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID), GGML_TYPE_I32);
            cb(cold_token_ids, "ffn_moe_cold_token_ids", il);
        }

        if (!cold_prefix_weighted_sum) {
            cold_weights = ggml_reshape_3d(ctx0, view_worklist_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT), 1, 1, capacity);
            cb(cold_weights, "ffn_moe_cold_weights_compact", il);
        }
    }

    if (hot_count != nullptr) {
        ggml_build_forward_expand(gf, hot_count);
    }
    if (cold_count != nullptr) {
        ggml_build_forward_expand(gf, cold_count);
    }
    ggml_build_forward_expand(gf, hot_ids);
    if (hot_src_slots != nullptr) {
        ggml_build_forward_expand(gf, hot_src_slots);
    }
    if (hot_token_ids != nullptr) {
        ggml_build_forward_expand(gf, hot_token_ids);
    }
    ggml_build_forward_expand(gf, hot_weights);
    if (hot_expert_ids != nullptr) {
        ggml_build_forward_expand(gf, hot_expert_ids);
    }
    if (cold_ids != nullptr) {
        ggml_build_forward_expand(gf, cold_ids);
        if (cold_src_slots != nullptr) {
            ggml_build_forward_expand(gf, cold_src_slots);
        }
        if (cold_token_ids != nullptr) {
            ggml_build_forward_expand(gf, cold_token_ids);
        }
        if (cold_weights != nullptr) {
            ggml_build_forward_expand(gf, cold_weights);
        }
    }

    ggml_tensor * hot_inputs = repeat_hot_input
        ? ggml_repeat_4d(ctx0, cur, n_embd, capacity, 1, 1)
        : ggml_get_rows(ctx0, cur, hot_token_ids);
    cb(hot_inputs, "ffn_moe_hot_inputs", il);

    const uint32_t hot_mul_mat_id_flags = llama_moe_hot_cache_graph_tweaks::hot_dummy_padding()
        ? LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_NONE
        : LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_ALLOW_NEGATIVE_IDS;

    const auto merge_compact_slots = [&](
            ggml_tensor * branch_out,
            ggml_backend_t branch_backend,
            ggml_tensor * prefix_count,
            ggml_tensor * prefix_worklist,
            bool prefix_weighted,
            const char * name) {
        if (prefix_count != nullptr && branch_backend != nullptr && n_tokens == 1 &&
                llama_moe_hot_cache_graph_tweaks::cold_prefix_sum()) {
            ggml_tensor * merged_shape = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
            ggml_backend_sched_set_tensor_backend(sched, merged_shape, branch_backend);

            ggml_tensor * merged = ggml_map_custom3(
                    ctx0,
                    merged_shape,
                    branch_out,
                    prefix_weighted ? prefix_worklist : prefix_count,
                    prefix_weighted
                        ? llama_qwen35moe_hot_cache_sum_weighted_prefix_rows_op
                        : llama_qwen35moe_hot_cache_sum_prefix_rows_op,
                    1,
                    nullptr);
            ggml_backend_sched_set_tensor_backend(sched, merged, branch_backend);
            cb(merged, name, il);
            ggml_build_forward_expand(gf, merged);
            return merged;
        }

        ggml_tensor * merged = ggml_reshape_3d(ctx0, branch_out, n_embd, capacity, 1);
        if (branch_backend != nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, merged, branch_backend);
        }
        merged = ggml_permute(ctx0, merged, 1, 0, 2, 3);
        if (branch_backend != nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, merged, branch_backend);
        }
        if (!llama_moe_hot_cache_graph_tweaks::decode_strided_sum_rows()) {
            merged = ggml_cont(ctx0, merged);
            if (branch_backend != nullptr) {
                ggml_backend_sched_set_tensor_backend(sched, merged, branch_backend);
            }
        }
        merged = ggml_sum_rows(ctx0, merged);
        if (branch_backend != nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, merged, branch_backend);
        }
        merged = ggml_reshape_2d(ctx0, merged, n_embd, n_tokens);
        if (branch_backend != nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, merged, branch_backend);
        }
        cb(merged, name, il);
        ggml_build_forward_expand(gf, merged);
        return merged;
    };

    const auto reduce_slots_in_branch = [&](
            ggml_tensor * slots,
            ggml_backend_t branch_backend,
            const char * name) {
        ggml_tensor * reduced = ggml_permute(ctx0, slots, 1, 0, 2, 3);
        if (branch_backend != nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, reduced, branch_backend);
        }
        if (!llama_moe_hot_cache_graph_tweaks::decode_strided_sum_rows()) {
            reduced = ggml_cont(ctx0, reduced);
            if (branch_backend != nullptr) {
                ggml_backend_sched_set_tensor_backend(sched, reduced, branch_backend);
            }
        }
        reduced = ggml_sum_rows(ctx0, reduced);
        if (branch_backend != nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, reduced, branch_backend);
        }
        reduced = ggml_reshape_2d(ctx0, reduced, n_embd, n_tokens);
        if (branch_backend != nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, reduced, branch_backend);
        }
        cb(reduced, name, il);
        ggml_build_forward_expand(gf, reduced);
        return reduced;
    };

    ggml_tensor * hot_out = llama_moe_hot_cache_build_moe_ffn_with_ids(*this,
            hot_inputs,
            hot_ids,
            hot_weights,
            cache.ffn_up_exps,
            cache.ffn_gate_exps,
            cache.ffn_down_exps,
            cache.n_hot + 1,
            1,
            LLM_FFN_SILU,
            il,
            cache.ffn_gate_up_exps,
            cache.ffn_up_exps_s,
            cache.ffn_gate_exps_s,
            cache.ffn_down_exps_s,
            hot_mul_mat_id_flags,
            "hot");
    cb(hot_out, "ffn_moe_hot_out", il);

    ggml_tensor * hot_slots = nullptr;
    if (decode_direct_merge) {
        hot_slots = merge_compact_slots(hot_out, nullptr, nullptr, nullptr, false, "ffn_moe_hot_slots");
    } else {
        hot_slots = ggml_scale(ctx0, hot_inputs, 0.0f);
        hot_slots = ggml_pad(ctx0, hot_slots, 0, 1, 0, 0);
        hot_slots = ggml_set_rows(ctx0, hot_slots, hot_out, hot_src_slots);
        hot_slots = ggml_view_2d(ctx0, hot_slots, n_embd, capacity, hot_slots->nb[1], 0);
        hot_slots = ggml_reshape_3d(ctx0, hot_slots, n_embd, n_moe_slots, n_tokens);
        cb(hot_slots, "ffn_moe_hot_slots", il);
        ggml_build_forward_expand(gf, hot_slots);
    }
    ggml_tensor * hot_parallel_output = hot_slots;
    if (branch_reduce_merge) {
        hot_parallel_output = reduce_slots_in_branch(hot_slots, nullptr, "ffn_moe_hot_slots_reduced");
    }

    ggml_tensor * out_slots = hot_slots;

    if (cache.n_hot != cache.n_expert) {
        ggml_backend_t cold_branch_backend = cparams.warmup ? nullptr : backend_cpu;

        ggml_tensor * cold_inputs = nullptr;
        if (cold_first_row_input) {
            ggml_tensor * cold_inputs_shape = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, capacity);
            ggml_backend_sched_set_tensor_backend(sched, cold_inputs_shape, cold_branch_backend);
            cold_inputs = ggml_map_custom2(
                    ctx0,
                    cold_inputs_shape,
                    cur,
                    llama_qwen35moe_hot_cache_first_row_input_op,
                    1,
                    nullptr);
        } else {
            cold_inputs = ggml_get_rows(ctx0, cur, cold_token_ids);
        }
        if (cold_shared_input_row && !cold_first_row_input) {
            const int32_t get_rows_flags = GGML_GET_ROWS_FLAG_FIRST_ROW_ONLY;
            memcpy(cold_inputs->op_params, &get_rows_flags, sizeof(get_rows_flags));
        }
        cb(cold_inputs, "ffn_moe_cold_inputs", il);
        // Keep the cold lane on CPU. Otherwise this gather inherits the CUDA activation
        // backend and the hot/cold parallel region resolves both lanes to CUDA.
        if (cold_branch_backend != nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, cold_inputs, cold_branch_backend);
        }

        const uint32_t cold_mul_mat_id_flags = LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_ALLOW_NEGATIVE_IDS |
            (cold_shared_input_row ? LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_SHARED_INPUT_ROW : LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_NONE) |
            (cold_prefix_merge ? LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_SKIP_NEGATIVE_ID_OUTPUT_ZERO : LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_NONE);

        ggml_tensor * cold_out = llama_moe_hot_cache_build_moe_ffn_with_ids(*this,
                cold_inputs,
                cold_ids,
                cold_weights,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                n_expert,
                1,
                LLM_FFN_SILU,
                il,
                layer.ffn_gate_up_exps,
                layer.ffn_up_exps_s,
                layer.ffn_gate_exps_s,
                layer.ffn_down_exps_s,
                cold_mul_mat_id_flags,
                "cold",
                cold_branch_backend,
                !cold_prefix_weighted_sum);
        cb(cold_out, "ffn_moe_cold_out", il);

        ggml_tensor * cold_slots = nullptr;
        if (decode_direct_merge) {
            cold_slots = merge_compact_slots(cold_out, cold_branch_backend, cold_count, worklist, cold_prefix_weighted_sum, "ffn_moe_cold_slots");
        } else {
            cold_slots = ggml_scale(ctx0, cold_inputs, 0.0f);
            if (cold_branch_backend != nullptr) {
                ggml_backend_sched_set_tensor_backend(sched, cold_slots, cold_branch_backend);
            }
            cold_slots = ggml_pad(ctx0, cold_slots, 0, 1, 0, 0);
            if (cold_branch_backend != nullptr) {
                ggml_backend_sched_set_tensor_backend(sched, cold_slots, cold_branch_backend);
            }
            cold_slots = ggml_set_rows(ctx0, cold_slots, cold_out, cold_src_slots);
            if (cold_branch_backend != nullptr) {
                ggml_backend_sched_set_tensor_backend(sched, cold_slots, cold_branch_backend);
            }
            cold_slots = ggml_view_2d(ctx0, cold_slots, n_embd, capacity, cold_slots->nb[1], 0);
            if (cold_branch_backend != nullptr) {
                ggml_backend_sched_set_tensor_backend(sched, cold_slots, cold_branch_backend);
            }
            cold_slots = ggml_reshape_3d(ctx0, cold_slots, n_embd, n_moe_slots, n_tokens);
            cb(cold_slots, "ffn_moe_cold_slots", il);
            if (cold_branch_backend != nullptr) {
                ggml_backend_sched_set_tensor_backend(sched, cold_slots, cold_branch_backend);
            }
            ggml_build_forward_expand(gf, cold_slots);
        }

        ggml_tensor * cold_parallel_output = cold_slots;
        if (branch_reduce_merge) {
            cold_parallel_output = reduce_slots_in_branch(cold_slots, cold_branch_backend, "ffn_moe_cold_slots_reduced");
        }

        out_slots = branch_reduce_merge
            ? ggml_add(ctx0, hot_parallel_output, cold_parallel_output)
            : ggml_add(ctx0, hot_slots, cold_slots);
        cb(out_slots, "ffn_moe_hot_cold_slots", il);

        if (parallel_mode != 0 && !cparams.warmup && annotate_parallel_region && hot_count != nullptr && cold_count != nullptr) {
            const uint32_t parallel_flags = branch_reduce_merge
                ? GGML_BACKEND_SCHED_MOE_HOT_CACHE_PARALLEL_FLAG_ALLOW_JOIN_BRIDGE
                : GGML_BACKEND_SCHED_MOE_HOT_CACHE_PARALLEL_FLAG_NONE;
            ggml_backend_sched_moe_hot_cache_parallel_region(
                    sched,
                    il,
                    parallel_mode,
                    capacity,
                    hot_count,
                    cold_count,
                    hot_inputs,
                    hot_parallel_output,
                    hot_parallel_output,
                    cold_inputs,
                    cold_parallel_output,
                    cold_parallel_output,
                    out_slots,
                    parallel_flags);
        }
    }

    ggml_tensor * out = nullptr;
    if (decode_direct_merge) {
        out = out_slots;
        cb(out, "ffn_moe_out", il);
        return out;
    }

    if (branch_reduce_merge) {
        out = out_slots;
        cb(out, "ffn_moe_out", il);
        return out;
    }

    if (n_moe_slots > 1 && llama_moe_hot_cache_graph_tweaks::merge_sum_rows()) {
        out = ggml_permute(ctx0, out_slots, 1, 0, 2, 3);
        out = ggml_cont(ctx0, out);
        out = ggml_sum_rows(ctx0, out);
        out = ggml_reshape_2d(ctx0, out, n_embd, n_tokens);
        ggml_build_forward_expand(gf, out);
    } else {
        ggml_tensor * slot_outputs[LLAMA_MAX_EXPERTS] = { nullptr };
        for (int64_t i = 0; i < n_moe_slots; ++i) {
            slot_outputs[i] = ggml_view_2d(ctx0, out_slots, n_embd, n_tokens, out_slots->nb[2], i*out_slots->nb[1]);
            ggml_build_forward_expand(gf, slot_outputs[i]);
        }

        out = slot_outputs[0];
        for (int64_t i = 1; i < n_moe_slots; ++i) {
            out = ggml_add(ctx0, out, slot_outputs[i]);
            ggml_build_forward_expand(gf, out);
        }

        if (n_moe_slots == 1) {
            out = ggml_cont(ctx0, out);
        }
    }

    cb(out, "ffn_moe_out", il);
    return out;
}

ggml_tensor * llama_model_gemma4::graph::build_layer_moe_hot(ggml_tensor * cur, ggml_tensor * logits, const int il) {
    return llama_moe_hot_cache_build_moe_hot_from_logits(*this, model, cur, logits, il, LLM_FFN_GELU);
}
