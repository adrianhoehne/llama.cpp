#pragma once

#include "ggml.h"

// Build a mul_mat_id node whose logical id rows are mapped to source rows.
// This keeps Hot-Cache PP from materializing duplicate cold input rows.
ggml_tensor * llama_moe_hot_cache_ggml_mul_mat_id_indirect(
        ggml_context * ctx,
        ggml_tensor  * as,
        ggml_tensor  * b,
        ggml_tensor  * ids,
        ggml_tensor  * row_ids);

// Build a mul_mat_id node that directly reduces expert-slot outputs into token
// rows. This is used by PP cold CPU lanes to avoid materializing cold_out.
ggml_tensor * llama_moe_hot_cache_ggml_mul_mat_id_token_reduce(
        ggml_context * ctx,
        ggml_tensor  * as,
        ggml_tensor  * b,
        ggml_tensor  * ids,
        ggml_tensor  * token_ids,
        ggml_tensor  * weights,
        int64_t        n_tokens);
