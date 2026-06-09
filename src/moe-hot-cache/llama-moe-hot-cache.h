#pragma once

#include "llama-moe-hot-cache-adapter.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

static constexpr size_t LLAMA_MOE_HOT_CACHE_MAX_EXPERT_LANES = 3;

struct llama_model;
struct llama_model_params;

struct llama_moe_hot_cache_entry {
    uint32_t layer = 0;
    uint32_t expert = 0;
    uint64_t hit_count = 0;
};

struct llama_moe_hot_cache_expert_observation {
    uint32_t expert = 0;
    uint64_t hot = 0;
    uint64_t cold = 0;
    uint64_t raw = 0;
};

struct llama_moe_hot_cache_layer_observation {
    uint32_t layer = 0;
    std::vector<llama_moe_hot_cache_expert_observation> experts;
    bool has_branch_counts = false;
    double cold_slots_per_call = 0.0;
    double parallel_join_wait_time_per_call_us = 0.0;
    double parallel_cold_lane_wall_time_per_call_us = 0.0;
    double parallel_hot_lane_wall_time_per_call_us = 0.0;
    double total_moe_time_per_call_us = 0.0;
    double wait_per_cold_slot_us = 0.0;
};

struct llama_moe_hot_cache_expert_size {
    uint32_t layer = 0;
    uint32_t expert = 0;
    size_t bytes = 0;
};

struct llama_moe_hot_cache_plan {
    std::vector<llama_moe_hot_cache_entry> observed;
    std::vector<llama_moe_hot_cache_expert_size> selected;
    size_t budget_bytes = 0;
    size_t used_bytes = 0;
};

enum class llama_moe_hot_cache_device_strategy {
    warm,
    hot_even,
};

struct llama_moe_hot_cache_multi_plan {
    std::vector<llama_moe_hot_cache_entry> observed;
    std::vector<llama_moe_hot_cache_plan> lanes;

    size_t selected_count() const;
    size_t used_bytes() const;
    size_t budget_bytes() const;
};

struct llama_moe_hot_cache_update_stats {
    bool active = false;
    double update_rate = 0.0;
    double hit_rate = 0.0;
    uint64_t hot_slots = 0;
    uint64_t cold_slots = 0;
    size_t hot_experts = 0;
    size_t candidates = 0;
    size_t max_exchange = 0;
    size_t exchanged = 0;
    size_t layers_changed = 0;
};

enum class llama_moe_hot_cache_weighting_mode {
    pressure,
    smooth_pressure,
    time,
    balanced,
    flat,
};

struct llama_moe_hot_cache_weighting_config {
    llama_moe_hot_cache_weighting_mode mode = llama_moe_hot_cache_weighting_mode::flat;
    double layer_curve = 0.5;
};

using llama_moe_hot_cache_qwen35moe_weighting_mode = llama_moe_hot_cache_weighting_mode;
using llama_moe_hot_cache_qwen35moe_weighting_config = llama_moe_hot_cache_weighting_config;

struct llama_moe_hot_cache_layer_lane {
    ggml_tensor * ffn_gate_up_exps = nullptr;
    ggml_tensor * ffn_gate_exps    = nullptr;
    ggml_tensor * ffn_up_exps      = nullptr;
    ggml_tensor * ffn_down_exps    = nullptr;
    ggml_tensor * ffn_gate_exps_s  = nullptr;
    ggml_tensor * ffn_up_exps_s    = nullptr;
    ggml_tensor * ffn_down_exps_s  = nullptr;

    ggml_tensor * hot_id_map = nullptr;
    ggml_tensor * hot_mask   = nullptr;
    ggml_tensor * cold_mask  = nullptr;

    std::vector<int32_t> hot_id_map_host;

    uint32_t n_hot = 0;
    uint32_t n_expert = 0;
    float expert_weights_scale = 0.0f;

    bool active() const {
        return n_hot > 0 && hot_id_map != nullptr && hot_mask != nullptr && cold_mask != nullptr;
    }
};

struct llama_moe_hot_cache_layer : llama_moe_hot_cache_layer_lane {
    std::vector<llama_moe_hot_cache_layer_lane> lanes;
    std::vector<int32_t> expert_lane_map_host;

    bool multi_lane_active() const {
        size_t active_lanes = 0;
        for (const auto & lane : lanes) {
            active_lanes += lane.active() ? 1 : 0;
        }
        return active_lanes > 1;
    }

    bool active() const {
        if (llama_moe_hot_cache_layer_lane::active()) {
            return true;
        }
        for (const auto & lane : lanes) {
            if (lane.active()) {
                return true;
            }
        }
        return false;
    }
};

enum llama_moe_hot_cache_worklist_field : int32_t {
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID = 0,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_COUNT,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_COUNT,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT1_ID,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT1_SRC_SLOT,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT1_TOKEN_ID,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT1_WEIGHT,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT1_EXPERT_ID,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT1_COUNT,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT2_ID,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT2_SRC_SLOT,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT2_TOKEN_ID,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT2_WEIGHT,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT2_EXPERT_ID,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT2_COUNT,
    LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COUNT,
};

struct llama_moe_hot_cache {
    std::vector<llama_moe_hot_cache_layer> layers;
    std::vector<ggml_backend_dev_t> devices;
    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    bool active() const {
        for (const auto & layer : layers) {
            if (layer.active()) {
                return true;
            }
        }
        return false;
    }
};

struct llama_moe_hot_cache_weighting {
    static bool parse_mode(const std::string & name, llama_moe_hot_cache_weighting_mode & mode);
    static const char * mode_name(llama_moe_hot_cache_weighting_mode mode);
    static llama_moe_hot_cache_weighting_config default_config();

    static std::vector<llama_moe_hot_cache_entry> score_observations(
            const std::vector<llama_moe_hot_cache_layer_observation> & observations);
    static std::vector<llama_moe_hot_cache_entry> score_observations(
            const std::vector<llama_moe_hot_cache_layer_observation> & observations,
            const llama_moe_hot_cache_weighting_config & config);
};

struct llama_moe_hot_cache_qwen35moe_weighting {
    static bool parse_mode(const std::string & name, llama_moe_hot_cache_qwen35moe_weighting_mode & mode);
    static const char * mode_name(llama_moe_hot_cache_qwen35moe_weighting_mode mode);
    static llama_moe_hot_cache_qwen35moe_weighting_config default_config();

    static std::vector<llama_moe_hot_cache_entry> score_observations(
            const std::vector<llama_moe_hot_cache_layer_observation> & observations);
    static std::vector<llama_moe_hot_cache_entry> score_observations(
            const std::vector<llama_moe_hot_cache_layer_observation> & observations,
            const llama_moe_hot_cache_qwen35moe_weighting_config & config);
};

struct llama_moe_hot_cache_gemma4_weighting {
    static std::vector<llama_moe_hot_cache_entry> score_observations(
            const std::vector<llama_moe_hot_cache_layer_observation> & observations);
};

std::vector<llama_moe_hot_cache_layer_observation> llama_moe_hot_cache_parse_perf_json_observations(
        const std::string & json_str);

std::vector<llama_moe_hot_cache_entry> llama_moe_hot_cache_parse_perf_json(const std::string & json_str);

llama_moe_hot_cache_plan llama_moe_hot_cache_select(
        const std::vector<llama_moe_hot_cache_entry> & observed,
        const std::vector<llama_moe_hot_cache_expert_size> & sizes,
        size_t budget_bytes);

void llama_moe_hot_cache_init(llama_model & model, const llama_model_params & params, bool reserve_kv_cache = true);
void llama_moe_hot_cache_init_after_model_load(llama_model & model, const llama_model_params & params);
void llama_moe_hot_cache_init_after_context_memory(const llama_model & model);

LLAMA_API llama_moe_hot_cache_update_stats llama_moe_hot_cache_update_from_perf_json(
        llama_model & model,
        const std::string & json_str,
        double update_rate);

LLAMA_API llama_moe_hot_cache_update_stats llama_moe_hot_cache_apply_json(
        llama_model & model,
        const std::string & json_str);

bool llama_moe_hot_cache_layer_active(const llama_model & model, int il);
bool llama_moe_hot_cache_layer_active_for_graph(
        const llama_model & model,
        int il,
        llama_moe_hot_cache_graph_kind graph_kind);

enum class llama_moe_hot_cache_worklist_order {
    token_major,
    expert_major,
};

const char * llama_moe_hot_cache_worklist_order_name(llama_moe_hot_cache_worklist_order order);

void llama_moe_hot_cache_build_worklist(
        ggml_tensor * dst,
        const ggml_tensor * selected_experts,
        const ggml_tensor * weights,
        const llama_moe_hot_cache_layer & layer,
        int ith,
        int nth,
        llama_moe_hot_cache_worklist_order order = llama_moe_hot_cache_worklist_order::token_major);

void llama_moe_hot_cache_build_worklist_from_logits(
        ggml_tensor * dst,
        const ggml_tensor * logits,
        const llama_moe_hot_cache_layer & layer,
        int ith,
        int nth,
        llama_moe_hot_cache_worklist_order order = llama_moe_hot_cache_worklist_order::token_major);
