#include "llama-moe-hot-cache-ggml.h"

ggml_tensor * llama_moe_hot_cache_ggml_mul_mat_id_indirect(
        ggml_context * ctx,
        ggml_tensor  * as,
        ggml_tensor  * b,
        ggml_tensor  * ids,
        ggml_tensor  * row_ids) {
    GGML_ASSERT(!ggml_is_transposed(as));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(row_ids->type == GGML_TYPE_I32);

    GGML_ASSERT(as->ne[3] == 1); // one matrix per expert
    GGML_ASSERT(b->ne[3] == 1);
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1);
    GGML_ASSERT(row_ids->ne[0] == ids->ne[1]);
    GGML_ASSERT(row_ids->ne[1] == 1 && row_ids->ne[2] == 1 && row_ids->ne[3] == 1);
    GGML_ASSERT(as->ne[0] == b->ne[0]);
    GGML_ASSERT(ids->ne[0] % b->ne[1] == 0);

    const int64_t ne[4] = { as->ne[1], ids->ne[0], ids->ne[1], 1 };
    ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT_ID;
    result->src[0] = as;
    result->src[1] = b;
    result->src[2] = ids;
    result->src[3] = row_ids;

    return result;
}

ggml_tensor * llama_moe_hot_cache_ggml_mul_mat_id_token_reduce(
        ggml_context * ctx,
        ggml_tensor  * as,
        ggml_tensor  * b,
        ggml_tensor  * ids,
        ggml_tensor  * token_ids,
        ggml_tensor  * weights,
        int64_t        n_tokens) {
    GGML_ASSERT(!ggml_is_transposed(as));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(token_ids->type == GGML_TYPE_I32);
    GGML_ASSERT(weights == nullptr || weights->type == GGML_TYPE_F32);

    GGML_ASSERT(as->ne[3] == 1); // one matrix per expert
    GGML_ASSERT(b->ne[3] == 1);
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1);
    GGML_ASSERT(ids->ne[0] == 1);
    GGML_ASSERT(ids->ne[1] == b->ne[2]);
    GGML_ASSERT(token_ids->ne[0] == ids->ne[1]);
    GGML_ASSERT(token_ids->ne[1] == 1 && token_ids->ne[2] == 1 && token_ids->ne[3] == 1);
    GGML_ASSERT(weights == nullptr || weights->ne[0] == 1);
    GGML_ASSERT(weights == nullptr || weights->ne[1] == 1);
    GGML_ASSERT(weights == nullptr || weights->ne[2] == ids->ne[1]);
    GGML_ASSERT(as->ne[0] == b->ne[0]);
    GGML_ASSERT(n_tokens > 0);

    const int64_t ne[4] = { as->ne[1], n_tokens, 1, 1 };
    ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT_ID;
    result->src[0] = as;
    result->src[1] = b;
    result->src[2] = ids;
    result->src[4] = token_ids;
    result->src[5] = weights;

    return result;
}
