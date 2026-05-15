#pragma once

#include "ggml-backend.h"

#include <cstdint>

struct ggml_tensor;
struct llama_context;

bool llama_moe_layer_perf_is_enabled(const llama_context * ctx);
bool llama_moe_layer_perf_eval_callback(ggml_tensor * t, bool ask, void * user_data);

void llama_moe_layer_perf_begin(uint32_t n_layer, uint32_t n_expert, uint32_t n_expert_used);
void llama_moe_layer_perf_end();
void llama_moe_layer_perf_reset();
bool llama_moe_layer_perf_has_data();
void llama_moe_layer_perf_collect_parallel_metrics(ggml_backend_sched_t sched);
bool llama_moe_layer_perf_graph_compute_begin(llama_context * ctx, ggml_backend_sched_t sched);
void llama_moe_layer_perf_graph_compute_end(llama_context * ctx, ggml_backend_sched_t sched);
