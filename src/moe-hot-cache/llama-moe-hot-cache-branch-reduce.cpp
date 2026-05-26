#include "llama-moe-hot-cache-branch-reduce.h"

#include "llama-moe-hot-cache.h"
#include "llama-graph.h"

#include <algorithm>
#include <cstring>

namespace {

static float worklist_field_value(const ggml_tensor * worklist, int32_t field, int64_t slot) {
    const char * row = (const char *) worklist->data + field*worklist->nb[1];
    return *(const float *) (row + slot*worklist->nb[0]);
}

static void llama_moe_hot_cache_compact_cold_reduce_op(
        ggml_tensor * dst,
        const ggml_tensor * shape,
        const ggml_tensor * branch_out,
        const ggml_tensor * worklist,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(shape);
    GGML_UNUSED(userdata);

    llama_moe_hot_cache_reduce_cold_token_rows(dst, branch_out, worklist, ith, nth);
}

} // namespace

void llama_moe_hot_cache_reduce_cold_token_rows(
        ggml_tensor * dst,
        const ggml_tensor * branch_out,
        const ggml_tensor * worklist,
        int ith,
        int nth) {
    GGML_ASSERT(dst != nullptr);
    GGML_ASSERT(branch_out != nullptr);
    GGML_ASSERT(worklist != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(branch_out->type == GGML_TYPE_F32);
    GGML_ASSERT(worklist->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[0] == branch_out->ne[0]);
    GGML_ASSERT(dst->ne[1] >= 1);
    GGML_ASSERT(worklist->ne[1] == LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COUNT);
    GGML_ASSERT(worklist->ne[0] >= branch_out->ne[1]);
    GGML_ASSERT(dst->nb[0] == sizeof(float));
    GGML_ASSERT(branch_out->nb[0] == sizeof(float));
    GGML_ASSERT(worklist->nb[0] == sizeof(float));

    const int64_t n_embd = dst->ne[0];
    const int64_t n_tokens = dst->ne[1];
    const int64_t capacity = branch_out->ne[1];
    const float count_f = worklist_field_value(worklist, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_COUNT, 0);
    int64_t n_prefix = count_f > 0.0f ? (int64_t) (count_f + 0.5f) : 0;
    n_prefix = std::min<int64_t>(n_prefix, capacity);

    const int64_t dr = (n_embd + nth - 1)/nth;
    const int64_t i0 = dr*ith;
    const int64_t i1 = std::min<int64_t>(i0 + dr, n_embd);
    const int64_t n_local_embd = i1 - i0;
    const size_t local_row_bytes = size_t(n_local_embd)*sizeof(float);

    if (n_local_embd <= 0) {
        return;
    }

    for (int64_t token = 0; token < n_tokens; ++token) {
        float * dst_row = (float *) ((char *) dst->data + i0*dst->nb[0] + token*dst->nb[1]);
        std::memset(dst_row, 0, local_row_bytes);
    }

    for (int64_t slot = 0; slot < n_prefix; ++slot) {
        const float token_f = worklist_field_value(worklist, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, slot);
        const int64_t token = token_f >= 0.0f ? (int64_t) (token_f + 0.5f) : -1;
        if (token < 0 || token >= n_tokens) {
            continue;
        }

        float * dst_row = (float *) ((char *) dst->data + i0*dst->nb[0] + token*dst->nb[1]);
        const float * src_row = (const float *) ((const char *) branch_out->data + i0*branch_out->nb[0] + slot*branch_out->nb[1]);
        for (int64_t i = 0; i < n_local_embd; ++i) {
            dst_row[i] += src_row[i];
        }
    }
}

ggml_tensor * llama_moe_hot_cache_build_compact_cold_reduce(
        const llm_graph_context & graph,
        ggml_tensor * branch_out,
        ggml_tensor * worklist,
        ggml_backend_t backend,
        int n_tasks,
        const char * name,
        int il,
        int64_t n_tokens) {
    GGML_ASSERT(branch_out != nullptr);
    GGML_ASSERT(worklist != nullptr);
    GGML_ASSERT(n_tokens > 1);

    ggml_tensor * shape = ggml_new_tensor_2d(graph.ctx0, GGML_TYPE_F32, branch_out->ne[0], n_tokens);
    if (backend != nullptr) {
        ggml_backend_sched_set_tensor_backend(graph.sched, shape, backend);
    }

    ggml_tensor * reduced = ggml_map_custom3(
            graph.ctx0,
            shape,
            branch_out,
            worklist,
            llama_moe_hot_cache_compact_cold_reduce_op,
            std::max(1, n_tasks),
            nullptr);
    if (backend != nullptr) {
        ggml_backend_sched_set_tensor_backend(graph.sched, reduced, backend);
    }
    graph.cb(reduced, name, il);
    ggml_build_forward_expand(graph.gf, reduced);
    return reduced;
}
