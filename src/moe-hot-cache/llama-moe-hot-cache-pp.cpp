#include "llama-moe-hot-cache-pp.h"

#include <cstdlib>
#include <cstring>

namespace {

enum class pp_reduce_merge_mode {
    off,
    on,
    automatic,
};

enum class pp_bypass_mode {
    off,
    on,
    threshold,
};

static bool env_is_false(const char * value) {
    return value == nullptr || value[0] == '\0' ||
           std::strcmp(value, "0") == 0 ||
           std::strcmp(value, "off") == 0 ||
           std::strcmp(value, "false") == 0;
}

static pp_reduce_merge_mode pp_reduce_merge_mode_value() {
    const char * env = std::getenv("LLAMA_MOE_HOT_CACHE_PP_REDUCE_MERGE");
    if (env_is_false(env)) {
        return pp_reduce_merge_mode::off;
    }
    if (std::strcmp(env, "auto") == 0) {
        return pp_reduce_merge_mode::automatic;
    }
    return pp_reduce_merge_mode::on;
}

static llama_moe_hot_cache_worklist_order pp_worklist_order_mode() {
    const char * env = std::getenv("LLAMA_MOE_HOT_CACHE_PP_WORKLIST_ORDER");
    if (env == nullptr || env[0] == '\0' || std::strcmp(env, "auto") == 0) {
        return llama_moe_hot_cache_worklist_order::expert_major;
    }
    if (std::strcmp(env, "expert") == 0 ||
            std::strcmp(env, "expert-major") == 0 ||
            std::strcmp(env, "1") == 0 ||
            std::strcmp(env, "on") == 0 ||
            std::strcmp(env, "true") == 0) {
        return llama_moe_hot_cache_worklist_order::expert_major;
    }
    return llama_moe_hot_cache_worklist_order::token_major;
}

static bool pp_compact_cold_reduce_mode() {
    const char * env = std::getenv("LLAMA_MOE_HOT_CACHE_PP_COMPACT_COLD_REDUCE");
    if (env == nullptr || env[0] == '\0') {
        return true;
    }

    return !env_is_false(env);
}

static pp_bypass_mode pp_bypass_hot_cache_mode() {
    const char * env = std::getenv("LLAMA_MOE_HOT_CACHE_PP_BYPASS");
    if (env == nullptr || env[0] == '\0' || std::strcmp(env, "auto") == 0 || std::strcmp(env, "threshold") == 0) {
        return pp_bypass_mode::threshold;
    }
    if (env_is_false(env)) {
        return pp_bypass_mode::off;
    }

    return pp_bypass_mode::on;
}

static int64_t pp_bypass_min_tokens(int64_t default_min_tokens) {
    const char * env = std::getenv("LLAMA_MOE_HOT_CACHE_PP_BYPASS_MIN_TOKENS");
    if (env == nullptr || env[0] == '\0') {
        return default_min_tokens;
    }

    char * end = nullptr;
    const long long parsed = std::strtoll(env, &end, 10);
    if (end == env || parsed < 0) {
        return default_min_tokens;
    }

    return (int64_t) parsed;
}

static double pp_bypass_min_hot_expert_ratio(double default_min_hot_expert_ratio) {
    const char * env = std::getenv("LLAMA_MOE_HOT_CACHE_PP_MIN_HOT_EXPERT_RATIO");
    if (env == nullptr || env[0] == '\0') {
        return default_min_hot_expert_ratio;
    }

    char * end = nullptr;
    const double parsed = std::strtod(env, &end);
    if (end == env || parsed < 0.0) {
        return default_min_hot_expert_ratio;
    }

    return parsed > 1.0 ? 1.0 : parsed;
}

static llama_moe_hot_cache_graph_phase graph_phase_from_llm(
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

} // namespace

llama_moe_hot_cache_pp_execution_plan llama_moe_hot_cache_pp_policy::build(
        llama_moe_hot_cache_graph_phase phase,
        int64_t n_tokens,
        int64_t capacity,
        bool has_cold_lane,
        int64_t n_moe_slots,
        const llama_moe_hot_cache_graph_profile & profile) {
    llama_moe_hot_cache_pp_execution_plan plan;
    plan.phase = phase;
    plan.worklist_order = worklist_order(phase);

    if (phase != llama_moe_hot_cache_graph_phase::warmup && has_cold_lane && n_moe_slots > 1) {
        if (phase == llama_moe_hot_cache_graph_phase::decode) {
            plan.branch_reduce_merge = n_tokens == 1 && profile.branch_reduce_merge;
        } else if (phase == llama_moe_hot_cache_graph_phase::prompt_processing) {
            plan.branch_reduce_merge = n_tokens > 1 && reduce_merge_enabled(n_tokens, capacity);
        }
    }
    plan.compact_cold_reduce =
        plan.branch_reduce_merge &&
        plan.phase == llama_moe_hot_cache_graph_phase::prompt_processing &&
        compact_cold_reduce_enabled(phase, n_tokens);

    return plan;
}

bool llama_moe_hot_cache_pp_policy::reduce_merge_enabled(int64_t n_tokens, int64_t capacity) {
    switch (pp_reduce_merge_mode_value()) {
        case pp_reduce_merge_mode::off:
            return false;
        case pp_reduce_merge_mode::on:
            return true;
        case pp_reduce_merge_mode::automatic:
            return n_tokens >= 32 && capacity >= 64;
    }

    return false;
}

bool llama_moe_hot_cache_pp_policy::compact_cold_reduce_enabled(llama_moe_hot_cache_graph_phase phase, int64_t n_tokens) {
    if (phase != llama_moe_hot_cache_graph_phase::prompt_processing || n_tokens <= 1) {
        return false;
    }

    return pp_compact_cold_reduce_mode();
}

llama_moe_hot_cache_worklist_order llama_moe_hot_cache_pp_policy::worklist_order(llama_moe_hot_cache_graph_phase phase) {
    if (phase != llama_moe_hot_cache_graph_phase::prompt_processing) {
        return llama_moe_hot_cache_worklist_order::token_major;
    }

    return pp_worklist_order_mode();
}

bool llama_moe_hot_cache_pp_policy::bypass_hot_cache_for_prompt_processing(
        llm_graph_phase phase,
        bool warmup,
        int64_t n_tokens,
        int64_t default_min_tokens) {
    return bypass_hot_cache_for_prompt_processing(
            phase,
            warmup,
            n_tokens,
            default_min_tokens,
            0,
            0,
            0.0);
}

bool llama_moe_hot_cache_pp_policy::bypass_hot_cache_for_prompt_processing(
        llm_graph_phase phase,
        bool warmup,
        int64_t n_tokens,
        int64_t default_min_tokens,
        uint32_t n_hot_experts,
        uint32_t n_total_experts,
        double default_min_hot_expert_ratio) {
    if (graph_phase_from_llm(phase, warmup, n_tokens) != llama_moe_hot_cache_graph_phase::prompt_processing) {
        return false;
    }

    switch (pp_bypass_hot_cache_mode()) {
        case pp_bypass_mode::off:
            return false;
        case pp_bypass_mode::on:
            return true;
        case pp_bypass_mode::threshold: {
            const int64_t threshold = pp_bypass_min_tokens(default_min_tokens);
            if (threshold > 0 && n_tokens >= threshold) {
                return true;
            }

            const double min_hot_expert_ratio = pp_bypass_min_hot_expert_ratio(default_min_hot_expert_ratio);
            if (min_hot_expert_ratio > 0.0 && n_total_experts > 0) {
                const double hot_expert_ratio = double(n_hot_experts) / double(n_total_experts);
                return hot_expert_ratio < min_hot_expert_ratio;
            }
            return false;
        }
    }

    return false;
}

const char * llama_moe_hot_cache_graph_phase_name(llama_moe_hot_cache_graph_phase phase) {
    switch (phase) {
        case llama_moe_hot_cache_graph_phase::warmup:
            return "warmup";
        case llama_moe_hot_cache_graph_phase::decode:
            return "decode";
        case llama_moe_hot_cache_graph_phase::prompt_processing:
            return "prompt_processing";
    }

    return "unknown";
}
