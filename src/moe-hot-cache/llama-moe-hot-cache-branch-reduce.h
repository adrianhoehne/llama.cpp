#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>

struct llm_graph_context;

void llama_moe_hot_cache_reduce_cold_token_rows(
        ggml_tensor * dst,
        const ggml_tensor * branch_out,
        const ggml_tensor * worklist,
        int ith,
        int nth);

ggml_tensor * llama_moe_hot_cache_build_compact_cold_reduce(
        const llm_graph_context & graph,
        ggml_tensor * branch_out,
        ggml_tensor * worklist,
        ggml_backend_t backend,
        int n_tasks,
        const char * name,
        int il,
        int64_t n_tokens);
