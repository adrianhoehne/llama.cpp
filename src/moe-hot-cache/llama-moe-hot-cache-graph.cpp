#include "llama-moe-hot-cache.h"
#include "llama-moe-hot-cache-adapter.h"
#include "llama-moe-hot-cache-branch-reduce.h"
#include "llama-moe-hot-cache-perf.h"
#include "llama-moe-hot-cache-pp.h"
#include "ggml-backend-moe-hot-cache.h"
#include "models/models.h"

#include <cassert>
#include <cstdio>
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

static const llama_moe_hot_cache_model_adapter & llama_moe_hot_cache_require_model_adapter(
        llm_arch arch,
        llama_moe_hot_cache_graph_kind graph_kind) {
    const llama_moe_hot_cache_model_adapter * adapter = llama_moe_hot_cache_find_model_adapter(arch, graph_kind);
    GGML_ASSERT(adapter != nullptr);
    return *adapter;
}

static llama_moe_hot_cache_graph_phase llama_moe_hot_cache_graph_phase_from_llm(
        llm_graph_phase phase,
        bool warmup,
        int64_t n_tokens) {
    if (warmup || phase == LLM_GRAPH_PHASE_WARMUP) {
        return llama_moe_hot_cache_graph_phase::warmup;
    }

    switch (phase) {
        case LLM_GRAPH_PHASE_PROMPT_PROCESSING:
            return llama_moe_hot_cache_graph_phase::prompt_processing;
        case LLM_GRAPH_PHASE_DECODE:
            return llama_moe_hot_cache_graph_phase::decode;
        case LLM_GRAPH_PHASE_UNKNOWN:
            return n_tokens > 1
                ? llama_moe_hot_cache_graph_phase::prompt_processing
                : llama_moe_hot_cache_graph_phase::decode;
        case LLM_GRAPH_PHASE_WARMUP:
            return llama_moe_hot_cache_graph_phase::warmup;
    }

    return llama_moe_hot_cache_graph_phase::decode;
}

template<llama_moe_hot_cache_worklist_order order>
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
    llama_moe_hot_cache_build_worklist(dst, selected_experts, weights, *layer, ith, nth, order);
}

template<llama_moe_hot_cache_worklist_order order>
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
    llama_moe_hot_cache_build_worklist_from_logits(dst, logits, *layer, ith, nth, order);
}

static ggml_custom3_op_t llama_moe_hot_cache_select_worklist_op(llama_moe_hot_cache_worklist_order order) {
    return order == llama_moe_hot_cache_worklist_order::expert_major
        ? llama_qwen35moe_hot_cache_build_worklist_op<llama_moe_hot_cache_worklist_order::expert_major>
        : llama_qwen35moe_hot_cache_build_worklist_op<llama_moe_hot_cache_worklist_order::token_major>;
}

static ggml_custom2_op_t llama_moe_hot_cache_select_worklist_from_logits_op(llama_moe_hot_cache_worklist_order order) {
    return order == llama_moe_hot_cache_worklist_order::expert_major
        ? llama_qwen35moe_hot_cache_build_worklist_from_logits_op<llama_moe_hot_cache_worklist_order::expert_major>
        : llama_qwen35moe_hot_cache_build_worklist_from_logits_op<llama_moe_hot_cache_worklist_order::token_major>;
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
         ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,
         ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,
         ggml_tensor * down_exps_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                 int   il,
             ggml_tensor * gate_up_exps,
             ggml_tensor * gate_up_exps_b,
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
    ggml_tensor * selected_experts_safe_ids = selected_experts;
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

    const bool needs_safe_ids =
        up_exps_b != nullptr ||
        gate_exps_b != nullptr ||
        down_exps_b != nullptr ||
        gate_up_exps_b != nullptr ||
        up_exps_s != nullptr ||
        gate_exps_s != nullptr ||
        down_exps_s != nullptr;

    if ((flags & LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_ALLOW_NEGATIVE_IDS) && needs_safe_ids) {
        ggml_tensor * scale_ids_f32 = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);
        scale_ids_f32 = ggml_clamp(ctx0, scale_ids_f32, 0.0f, float(n_expert - 1));
        selected_experts_safe_ids = ggml_cast(ctx0, scale_ids_f32, GGML_TYPE_I32);
        cb_moe(selected_experts_safe_ids, "ffn_moe_scale_ids");
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

        if (gate_up_exps_b) {
            gate_up = ggml_add_id(ctx0, gate_up, gate_up_exps_b, selected_experts_safe_ids);
            cb_moe(gate_up, "ffn_moe_gate_up_biased");
        }

        if (up_exps_s) {
            ggml_tensor * s = ggml_reshape_3d(ctx0, up_exps_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, selected_experts_safe_ids);
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

        if (up_exps_b) {
            up = ggml_add_id(ctx0, up, up_exps_b, selected_experts_safe_ids);
            cb_moe(up, "ffn_moe_up_biased");
        }

        if (up_exps_s) {
            ggml_tensor * s = ggml_reshape_3d(ctx0, up_exps_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, selected_experts_safe_ids);
            up = ggml_mul(ctx0, up, s);
            cb_moe(up, "ffn_moe_up_scaled");
        }

        if (gate_exps) {
            cur = llama_moe_hot_cache_build_lora_mm_id(graph, gate_exps, cur, selected_experts, intermediate_flags);
            cb_moe(cur, "ffn_moe_gate");
        } else {
            cur = up;
        }

        if (gate_exps_b) {
            cur = ggml_add_id(ctx0, cur, gate_exps_b, selected_experts_safe_ids);
            cb_moe(cur, "ffn_moe_gate_biased");
        }

        if (gate_exps_s) {
            ggml_tensor * s = ggml_reshape_3d(ctx0, gate_exps_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, selected_experts_safe_ids);
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
        case LLM_FFN_SWIGLU_OAI_MOE:
            {
                constexpr float alpha = 1.702f;
                constexpr float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, up, alpha, limit);
                cb_moe(cur, "ffn_moe_swiglu_oai");
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

    if (down_exps_b) {
        experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts_safe_ids);
        cb_moe(experts, "ffn_moe_down_biased");
    }

    if (down_exps_s) {
        ggml_tensor * s = ggml_reshape_3d(ctx0, down_exps_s, 1, n_expert, 1);
        s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
        s = ggml_get_rows(ctx0, s, selected_experts_safe_ids);
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

static ggml_backend_t llama_moe_hot_cache_backend_for_dev(
        ggml_backend_sched_t sched,
        ggml_backend_dev_t dev) {
    if (sched == nullptr || dev == nullptr) {
        return nullptr;
    }

    const int n_backends = ggml_backend_sched_get_n_backends(sched);
    for (int i = 0; i < n_backends; ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        if (ggml_backend_get_device(backend) == dev) {
            return backend;
        }
    }

    return nullptr;
}

static ggml_backend_t llama_moe_hot_cache_primary_merge_backend(
        ggml_backend_sched_t sched,
        const llama_model & model,
        int il) {
    // Worker lanes are expert-only. The final lane join follows the normal
    // layer placement, which is the primary graph device for split-mode none.
    return llama_moe_hot_cache_backend_for_dev(sched, model.dev_layer(il));
}

static int32_t llama_moe_hot_cache_lane_id_field(size_t lane) {
    switch (lane) {
        case 0: return LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID;
        case 1: return LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT1_ID;
        case 2: return LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT2_ID;
    }
    GGML_ABORT("invalid MoE hot-cache lane");
}

static int32_t llama_moe_hot_cache_lane_weight_field(size_t lane) {
    switch (lane) {
        case 0: return LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT;
        case 1: return LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT1_WEIGHT;
        case 2: return LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT2_WEIGHT;
    }
    GGML_ABORT("invalid MoE hot-cache lane");
}

static int32_t llama_moe_hot_cache_lane_count_field(size_t lane) {
    switch (lane) {
        case 0: return LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_COUNT;
        case 1: return LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT1_COUNT;
        case 2: return LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT2_COUNT;
    }
    GGML_ABORT("invalid MoE hot-cache lane");
}

static bool llama_moe_hot_cache_layer_has_cold_experts(
        const llama_moe_hot_cache_layer & cache) {
    if (cache.expert_lane_map_host.empty()) {
        return cache.n_hot != cache.n_expert;
    }

    for (int32_t lane : cache.expert_lane_map_host) {
        if (lane < 0) {
            return true;
        }
    }

    return false;
}

static ggml_tensor * llama_moe_hot_cache_build_moe_hot_multi_from_logits(
        const llm_graph_context & graph,
        const llama_model & model,
             ggml_tensor * cur,
             ggml_tensor * logits,
                 int   il,
        const llama_moe_hot_cache_model_adapter & adapter) {
    ggml_context * ctx0 = graph.ctx0;
    ggml_cgraph * gf = graph.gf;
    ggml_backend_sched_t sched = graph.sched;
    const llama_hparams & hparams = graph.hparams;
    const llama_cparams & cparams = graph.cparams;
    const int64_t n_embd = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const int64_t n_expert = graph.n_expert;
    const int64_t n_moe_slots = cparams.warmup ? hparams.n_expert_used : graph.n_expert_used;
    const auto & layer = model.layers[il];
    const auto & cache = model.moe_hot_cache->layers[il];
    const auto profile = adapter.profile();

    GGML_ASSERT(adapter.graph_kind != llama_moe_hot_cache_graph_kind::none);
    GGML_ASSERT(!cache.lanes.empty());
    GGML_ASSERT(cache.lanes.size() <= LLAMA_MOE_HOT_CACHE_MAX_EXPERT_LANES);
    GGML_ASSERT(n_tokens == 1);
    GGML_ASSERT(n_moe_slots > 0);
    GGML_ASSERT(n_moe_slots <= LLAMA_MAX_EXPERTS);

    const int64_t capacity = n_moe_slots*n_tokens;
    ggml_tensor * worklist_shape = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, capacity, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COUNT);
    ggml_tensor * worklist = nullptr;

    const bool cpu_decode_routing =
        profile.cpu_decode_routing &&
        n_tokens >= 1 && n_tokens <= profile.cpu_decode_routing_max_tokens;
    if (cpu_decode_routing) {
        worklist = ggml_map_custom2(
                ctx0,
                worklist_shape,
                logits,
                llama_moe_hot_cache_select_worklist_from_logits_op(llama_moe_hot_cache_worklist_order::token_major),
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
                llama_moe_hot_cache_select_worklist_op(llama_moe_hot_cache_worklist_order::token_major),
                1,
                const_cast<llama_moe_hot_cache_layer *>(&cache));
    }
    graph.cb(worklist, "ffn_moe_worklist", il);

    const auto view_worklist_field = [&](int32_t field) {
        return ggml_view_1d(ctx0, worklist, capacity, field*worklist->nb[1]);
    };
    const auto view_worklist_count = [&](int32_t field) {
        return ggml_view_1d(ctx0, worklist, 1, field*worklist->nb[1]);
    };

    const auto merge_compact_slots = [&](
            ggml_tensor * branch_out,
            ggml_backend_t branch_backend,
            const char * name) {
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

    const uint32_t hot_mul_mat_id_flags = llama_moe_hot_cache_graph_tweaks::hot_dummy_padding()
        ? LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_NONE
        : LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_ALLOW_NEGATIVE_IDS;

    std::vector<ggml_tensor *> branch_outputs;
    branch_outputs.reserve(cache.lanes.size() + 1);

    for (size_t lane_index = 0; lane_index < cache.lanes.size(); ++lane_index) {
        const auto & lane = cache.lanes[lane_index];
        if (!lane.active()) {
            continue;
        }

        ggml_tensor * lane_count = view_worklist_count(llama_moe_hot_cache_lane_count_field(lane_index));
        graph.cb(lane_count, format("ffn_moe_hot%zu_count", lane_index).c_str(), il);
        ggml_build_forward_expand(gf, lane_count);

        ggml_tensor * lane_ids = ggml_cast(ctx0, view_worklist_field(llama_moe_hot_cache_lane_id_field(lane_index)), GGML_TYPE_I32);
        lane_ids = ggml_reshape_2d(ctx0, lane_ids, 1, capacity);
        graph.cb(lane_ids, format("ffn_moe_hot%zu_ids_compact", lane_index).c_str(), il);
        ggml_build_forward_expand(gf, lane_ids);

        ggml_tensor * lane_weights = ggml_reshape_3d(ctx0, view_worklist_field(llama_moe_hot_cache_lane_weight_field(lane_index)), 1, 1, capacity);
        graph.cb(lane_weights, format("ffn_moe_hot%zu_weights_compact", lane_index).c_str(), il);
        ggml_build_forward_expand(gf, lane_weights);

        ggml_backend_t lane_backend = nullptr;
        if (lane_index < model.moe_hot_cache->devices.size()) {
            lane_backend = llama_moe_hot_cache_backend_for_dev(sched, model.moe_hot_cache->devices[lane_index]);
        }

        ggml_tensor * lane_inputs = ggml_repeat_4d(ctx0, cur, n_embd, capacity, 1, 1);
        if (lane_backend != nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, lane_inputs, lane_backend);
        }
        graph.cb(lane_inputs, format("ffn_moe_hot%zu_inputs", lane_index).c_str(), il);

        const std::string branch_name = format("hot%zu", lane_index);
        ggml_tensor * lane_out = llama_moe_hot_cache_build_moe_ffn_with_ids(
                graph,
                lane_inputs,
                lane_ids,
                lane_weights,
                lane.ffn_up_exps,
                lane.ffn_up_exps_b,
                lane.ffn_gate_exps,
                lane.ffn_gate_exps_b,
                lane.ffn_down_exps,
                lane.ffn_down_exps_b,
                lane.n_hot + 1,
                1,
                adapter.ffn_op,
                il,
                lane.ffn_gate_up_exps,
                lane.ffn_gate_up_exps_b,
                lane.ffn_up_exps_s,
                lane.ffn_gate_exps_s,
                lane.ffn_down_exps_s,
                hot_mul_mat_id_flags,
                branch_name.c_str(),
                lane_backend);
        graph.cb(lane_out, format("ffn_moe_hot%zu_out", lane_index).c_str(), il);

        branch_outputs.push_back(merge_compact_slots(
                lane_out,
                lane_backend,
                format("ffn_moe_hot%zu_slots", lane_index).c_str()));
    }

    if (llama_moe_hot_cache_layer_has_cold_experts(cache)) {
        ggml_backend_t cold_backend = cparams.warmup ? nullptr : graph.backend_cpu;
        ggml_tensor * cold_ids = ggml_cast(ctx0, view_worklist_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID), GGML_TYPE_I32);
        cold_ids = ggml_reshape_2d(ctx0, cold_ids, 1, capacity);
        graph.cb(cold_ids, "ffn_moe_cold_ids_compact", il);
        ggml_build_forward_expand(gf, cold_ids);

        ggml_tensor * cold_weights = ggml_reshape_3d(ctx0, view_worklist_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT), 1, 1, capacity);
        graph.cb(cold_weights, "ffn_moe_cold_weights_compact", il);
        ggml_build_forward_expand(gf, cold_weights);

        ggml_tensor * cold_count = view_worklist_count(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_COUNT);
        graph.cb(cold_count, "ffn_moe_cold_count", il);
        ggml_build_forward_expand(gf, cold_count);

        ggml_tensor * cold_inputs = ggml_repeat_4d(ctx0, cur, n_embd, capacity, 1, 1);
        if (cold_backend != nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, cold_inputs, cold_backend);
        }
        graph.cb(cold_inputs, "ffn_moe_cold_inputs", il);

        const uint32_t cold_mul_mat_id_flags =
            LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_ALLOW_NEGATIVE_IDS |
            LLAMA_MOE_HOT_CACHE_MUL_MAT_ID_FLAG_SKIP_NEGATIVE_ID_OUTPUT_ZERO;
        ggml_tensor * cold_out = llama_moe_hot_cache_build_moe_ffn_with_ids(
                graph,
                cold_inputs,
                cold_ids,
                cold_weights,
                layer.ffn_up_exps,
                layer.ffn_up_exps_b,
                layer.ffn_gate_exps,
                layer.ffn_gate_exps_b,
                layer.ffn_down_exps,
                layer.ffn_down_exps_b,
                n_expert,
                1,
                adapter.ffn_op,
                il,
                layer.ffn_gate_up_exps,
                layer.ffn_gate_up_exps_b,
                layer.ffn_up_exps_s,
                layer.ffn_gate_exps_s,
                layer.ffn_down_exps_s,
                cold_mul_mat_id_flags,
                "cold",
                cold_backend);
        graph.cb(cold_out, "ffn_moe_cold_out", il);
        branch_outputs.push_back(merge_compact_slots(cold_out, cold_backend, "ffn_moe_cold_slots"));
    }

    GGML_ASSERT(!branch_outputs.empty());

    ggml_backend_t merge_backend = llama_moe_hot_cache_primary_merge_backend(sched, model, il);
    ggml_tensor * out = branch_outputs[0];
    for (size_t i = 1; i < branch_outputs.size(); ++i) {
        out = ggml_add(ctx0, out, branch_outputs[i]);
        if (merge_backend != nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, out, merge_backend);
        }
        ggml_build_forward_expand(gf, out);
    }

    if (merge_backend != nullptr) {
        ggml_backend_sched_set_tensor_backend(sched, out, merge_backend);
    }
    graph.cb(out, "ffn_moe_out", il);
    return out;
}

static ggml_tensor * llama_moe_hot_cache_build_moe_hot_from_logits(
        const llm_graph_context & graph,
        const llama_model & model,
             ggml_tensor * cur,
             ggml_tensor * logits,
                 int   il,
        const llama_moe_hot_cache_model_adapter & adapter) {
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
    const llama_moe_hot_cache_graph_profile profile = adapter.profile();
    const llama_moe_hot_cache_graph_phase graph_phase =
        llama_moe_hot_cache_graph_phase_from_llm(graph.gphase, cparams.warmup, n_tokens);
    const bool is_decode_phase = graph_phase == llama_moe_hot_cache_graph_phase::decode;
    const bool is_warmup_phase = graph_phase == llama_moe_hot_cache_graph_phase::warmup;

    GGML_ASSERT(adapter.graph_kind == llama_moe_hot_cache_graph_kind::logits);
    if (!cache.lanes.empty()) {
        return llama_moe_hot_cache_build_moe_hot_multi_from_logits(graph, model, cur, logits, il, adapter);
    }
    GGML_ASSERT(!cache.hot_id_map_host.empty());
    GGML_ASSERT(n_moe_slots > 0);
    GGML_ASSERT(n_moe_slots <= LLAMA_MAX_EXPERTS);

    const int64_t capacity = n_moe_slots*n_tokens;
    const llama_moe_hot_cache_pp_execution_plan pp_plan = llama_moe_hot_cache_pp_policy::build(
            graph_phase,
            n_tokens,
            capacity,
            cache.n_hot != cache.n_expert,
            n_moe_slots,
            profile);
    const int parallel_mode = llama_moe_hot_cache_graph_tweaks::parallel_mode();
    const int64_t parallel_min_slots = llama_moe_hot_cache_graph_tweaks::parallel_min_slots();
    const bool annotate_parallel_region =
        parallel_mode == 2 || parallel_min_slots == 0 || capacity >= parallel_min_slots;
    const bool decode_direct_merge =
        is_decode_phase && n_tokens == 1 && profile.decode_direct_merge;
    const bool cold_prefix_merge =
        cache.n_hot != cache.n_expert && decode_direct_merge && profile.cold_prefix_sum;
    const bool cold_prefix_weighted_sum =
        cold_prefix_merge && profile.cold_prefix_weighted_sum;
    const bool cold_shared_input_row =
        cache.n_hot != cache.n_expert && is_decode_phase && n_tokens == 1 && profile.shared_input_row;
    const bool cold_first_row_input =
        cold_shared_input_row && profile.cold_first_row_input;
    const bool perf_expert_counts = llama_moe_layer_perf_needs_expert_counts(cparams.no_perf);
    const bool compact_cold_reduce = pp_plan.compact_cold_reduce;
    const bool perf_or_parallel_counts =
        !is_warmup_phase && (perf_expert_counts || compact_cold_reduce || (parallel_mode != 0 && annotate_parallel_region));
    const bool repeat_hot_input =
        is_decode_phase && n_tokens == 1 && profile.decode_repeat_hot_input;
    const bool branch_reduce_merge = !decode_direct_merge && pp_plan.branch_reduce_merge;
    const bool use_compact_cold_reduce = branch_reduce_merge && compact_cold_reduce;

    ggml_tensor * worklist_shape = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, capacity, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COUNT);
    ggml_tensor * worklist = nullptr;
    const bool cpu_decode_routing =
        is_decode_phase && profile.cpu_decode_routing &&
        n_tokens >= 1 && n_tokens <= profile.cpu_decode_routing_max_tokens;
    if (cpu_decode_routing) {
        worklist = ggml_map_custom2(
                ctx0,
                worklist_shape,
                logits,
                llama_moe_hot_cache_select_worklist_from_logits_op(pp_plan.worklist_order),
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
                llama_moe_hot_cache_select_worklist_op(pp_plan.worklist_order),
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

        if (!decode_direct_merge && !use_compact_cold_reduce) {
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

            const int prefix_reduce_tasks = std::max<int>(1, std::min<int>(
                    profile.prefix_reduce_tasks_max,
                    cparams.n_threads));
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
            cache.ffn_up_exps_b,
            cache.ffn_gate_exps,
            cache.ffn_gate_exps_b,
            cache.ffn_down_exps,
            cache.ffn_down_exps_b,
            cache.n_hot + 1,
            1,
            adapter.ffn_op,
            il,
            cache.ffn_gate_up_exps,
            cache.ffn_gate_up_exps_b,
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
                layer.ffn_up_exps_b,
                layer.ffn_gate_exps,
                layer.ffn_gate_exps_b,
                layer.ffn_down_exps,
                layer.ffn_down_exps_b,
                n_expert,
                1,
                adapter.ffn_op,
                il,
                layer.ffn_gate_up_exps,
                layer.ffn_gate_up_exps_b,
                layer.ffn_up_exps_s,
                layer.ffn_gate_exps_s,
                layer.ffn_down_exps_s,
                cold_mul_mat_id_flags,
                "cold",
                cold_branch_backend,
                !cold_prefix_weighted_sum);
        graph.cb(cold_out, "ffn_moe_cold_out", il);

        ggml_tensor * cold_slots = nullptr;
        ggml_tensor * cold_parallel_output = nullptr;
        if (decode_direct_merge) {
            cold_slots = merge_compact_slots(cold_out, cold_branch_backend, cold_count, worklist, cold_prefix_weighted_sum, "ffn_moe_cold_slots");
            cold_parallel_output = cold_slots;
        } else if (use_compact_cold_reduce) {
            cold_parallel_output = llama_moe_hot_cache_build_compact_cold_reduce(
                    graph,
                    cold_out,
                    worklist,
                    cold_branch_backend,
                    cparams.n_threads,
                    "ffn_moe_cold_slots_reduced_compact",
                    il,
                    n_tokens);
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
            cold_parallel_output = cold_slots;
        }

        if (branch_reduce_merge && !use_compact_cold_reduce) {
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
    const llama_moe_hot_cache_model_adapter & adapter =
        llama_moe_hot_cache_require_model_adapter(model.arch, llama_moe_hot_cache_graph_kind::qwen35_ffn);
    const llama_moe_hot_cache_graph_profile profile = adapter.profile();

    const int64_t n_embd = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const int64_t n_moe_slots = cparams.warmup ? hparams.n_expert_used : n_expert_used;
    const auto & layer = model.layers[il];
    const auto & cache = model.moe_hot_cache->layers[il];
    const llama_moe_hot_cache_graph_phase graph_phase =
        llama_moe_hot_cache_graph_phase_from_llm(gphase, cparams.warmup, n_tokens);
    const bool is_decode_phase = graph_phase == llama_moe_hot_cache_graph_phase::decode;
    const bool is_warmup_phase = graph_phase == llama_moe_hot_cache_graph_phase::warmup;

    GGML_ASSERT(n_moe_slots > 0);
    GGML_ASSERT(n_moe_slots <= LLAMA_MAX_EXPERTS);

    ggml_tensor * logits = build_lora_mm(layer.ffn_gate_inp, cur);
    cb(logits, "ffn_moe_logits", il);

    if (!cache.lanes.empty()) {
        return llama_moe_hot_cache_build_moe_hot_multi_from_logits(*this, model, cur, logits, il, adapter);
    }

    GGML_ASSERT(!cache.hot_id_map_host.empty());

    const int64_t capacity = n_moe_slots*n_tokens;
    const llama_moe_hot_cache_pp_execution_plan pp_plan = llama_moe_hot_cache_pp_policy::build(
            graph_phase,
            n_tokens,
            capacity,
            cache.n_hot != cache.n_expert,
            n_moe_slots,
            profile);
    const int parallel_mode = llama_moe_hot_cache_graph_tweaks::parallel_mode();
    const int64_t parallel_min_slots = llama_moe_hot_cache_graph_tweaks::parallel_min_slots();
    const bool annotate_parallel_region =
        parallel_mode == 2 || parallel_min_slots == 0 || capacity >= parallel_min_slots;
    const bool decode_direct_merge =
        is_decode_phase && n_tokens == 1 && llama_moe_hot_cache_graph_tweaks::decode_direct_merge();
    const bool cold_prefix_merge =
        cache.n_hot != cache.n_expert && decode_direct_merge && llama_moe_hot_cache_graph_tweaks::cold_prefix_sum();
    const bool cold_prefix_weighted_sum =
        cold_prefix_merge && llama_moe_hot_cache_graph_tweaks::cold_prefix_weighted_sum();
    const bool cold_shared_input_row =
        cache.n_hot != cache.n_expert && decode_direct_merge && llama_moe_hot_cache_graph_tweaks::shared_input_row();
    const bool cold_first_row_input =
        cold_shared_input_row && llama_moe_hot_cache_graph_tweaks::cold_first_row_input();
    const bool perf_expert_counts = llama_moe_layer_perf_needs_expert_counts(cparams.no_perf);
    const bool compact_cold_reduce = pp_plan.compact_cold_reduce;
    const bool perf_or_parallel_counts =
        !is_warmup_phase && (perf_expert_counts || compact_cold_reduce || (parallel_mode != 0 && annotate_parallel_region));
    const bool repeat_hot_input =
        decode_direct_merge && llama_moe_hot_cache_graph_tweaks::decode_repeat_hot_input();
    const bool branch_reduce_merge = !decode_direct_merge && pp_plan.branch_reduce_merge;
    const bool use_compact_cold_reduce = branch_reduce_merge && compact_cold_reduce;

    ggml_tensor * worklist_shape = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, capacity, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COUNT);
    ggml_tensor * worklist = nullptr;
    if (is_decode_phase && n_tokens == 1 && llama_moe_hot_cache_graph_tweaks::cpu_decode_routing()) {
        worklist = ggml_map_custom2(
                ctx0,
                worklist_shape,
                logits,
                llama_moe_hot_cache_select_worklist_from_logits_op(pp_plan.worklist_order),
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
                llama_moe_hot_cache_select_worklist_op(pp_plan.worklist_order),
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

        if (!decode_direct_merge && !use_compact_cold_reduce) {
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
            nullptr,
            cache.ffn_gate_exps,
            nullptr,
            cache.ffn_down_exps,
            nullptr,
            cache.n_hot + 1,
            1,
            LLM_FFN_SILU,
            il,
            cache.ffn_gate_up_exps,
            nullptr,
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
                nullptr,
                layer.ffn_gate_exps,
                nullptr,
                layer.ffn_down_exps,
                nullptr,
                n_expert,
                1,
                LLM_FFN_SILU,
                il,
                layer.ffn_gate_up_exps,
                nullptr,
                layer.ffn_up_exps_s,
                layer.ffn_gate_exps_s,
                layer.ffn_down_exps_s,
                cold_mul_mat_id_flags,
                "cold",
                cold_branch_backend,
                !cold_prefix_weighted_sum);
        cb(cold_out, "ffn_moe_cold_out", il);

        ggml_tensor * cold_slots = nullptr;
        ggml_tensor * cold_parallel_output = nullptr;
        if (decode_direct_merge) {
            cold_slots = merge_compact_slots(cold_out, cold_branch_backend, cold_count, worklist, cold_prefix_weighted_sum, "ffn_moe_cold_slots");
            cold_parallel_output = cold_slots;
        } else if (use_compact_cold_reduce) {
            cold_parallel_output = llama_moe_hot_cache_build_compact_cold_reduce(
                    *this,
                    cold_out,
                    worklist,
                    cold_branch_backend,
                    cparams.n_threads,
                    "ffn_moe_cold_slots_reduced_compact",
                    il,
                    n_tokens);
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
            cold_parallel_output = cold_slots;
        }

        if (branch_reduce_merge && !use_compact_cold_reduce) {
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
    const llama_moe_hot_cache_model_adapter & adapter =
        llama_moe_hot_cache_require_model_adapter(model.arch, llama_moe_hot_cache_graph_kind::logits);
    return llama_moe_hot_cache_build_moe_hot_from_logits(*this, model, cur, logits, il, adapter);
}

ggml_tensor * llama_model_qwen3next::graph::build_layer_moe_hot(ggml_tensor * cur, ggml_tensor * logits, const int il) {
    const llama_moe_hot_cache_model_adapter & adapter =
        llama_moe_hot_cache_require_model_adapter(model.arch, llama_moe_hot_cache_graph_kind::logits);
    return llama_moe_hot_cache_build_moe_hot_from_logits(*this, model, cur, logits, il, adapter);
}

ggml_tensor * llama_model_openai_moe::graph::build_layer_moe_hot(ggml_tensor * cur, ggml_tensor * logits, const int il) {
    const llama_moe_hot_cache_model_adapter & adapter =
        llama_moe_hot_cache_require_model_adapter(model.arch, llama_moe_hot_cache_graph_kind::logits);
    return llama_moe_hot_cache_build_moe_hot_from_logits(*this, model, cur, logits, il, adapter);
}

template <bool iswa>
ggml_tensor * llama_model_mellum::graph<iswa>::build_layer_moe_hot(ggml_tensor * cur, ggml_tensor * logits, const int il) {
    const llama_moe_hot_cache_model_adapter & adapter =
        llama_moe_hot_cache_require_model_adapter(model.arch, llama_moe_hot_cache_graph_kind::logits);
    return llama_moe_hot_cache_build_moe_hot_from_logits(*this, model, cur, logits, il, adapter);
}

template ggml_tensor * llama_model_mellum::graph<false>::build_layer_moe_hot(ggml_tensor * cur, ggml_tensor * logits, int il);
template ggml_tensor * llama_model_mellum::graph<true>::build_layer_moe_hot(ggml_tensor * cur, ggml_tensor * logits, int il);
