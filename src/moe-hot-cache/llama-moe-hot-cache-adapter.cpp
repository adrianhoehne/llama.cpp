#include "llama-moe-hot-cache-adapter.h"
#include "llama-moe-hot-cache-pp.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace {

static bool env_enabled_by_default(const char * name) {
    const char * env = std::getenv(name);
    return env == nullptr || env[0] == '\0' ||
           (std::strcmp(env, "0") != 0 && std::strcmp(env, "off") != 0 && std::strcmp(env, "false") != 0);
}

static llama_moe_hot_cache_graph_profile qwen35_profile() {
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
        /* cpu_decode_routing_max_tokens = */ 1,
        /* prefix_reduce_tasks_max = */ llama_moe_hot_cache_graph_tweaks::prefix_reduce_tasks_max(),
    };
}

static llama_moe_hot_cache_graph_profile qwen3next_profile() {
    // Qwen3-Coder-Next uses the Qwen3Next MoE block plus a separate shared
    // expert path. It also reaches the MoE block with tiny multi-token batches
    // during decode, so keep routing on the compact CPU path for those batches
    // while leaving merge shortcuts at their single-token constraints.
    llama_moe_hot_cache_graph_profile profile = qwen35_profile();
    profile.cpu_decode_routing_max_tokens = 4;
    profile.prefix_reduce_tasks_max = 1;
    return profile;
}

static llama_moe_hot_cache_graph_profile gemma4_profile() {
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
        /* cpu_decode_routing_max_tokens = */ 1,
        /* prefix_reduce_tasks_max = */ llama_moe_hot_cache_graph_tweaks::prefix_reduce_tasks_max(),
    };
}

static llama_moe_hot_cache_graph_profile mellum_profile() {
    // Mellum uses a Qwen-style sparse SILU MoE block with plain router logits.
    // Keep the profile conservative until Mellum-specific PP shortcuts are measured.
    return qwen35_profile();
}

static llama_moe_hot_cache_graph_profile openai_moe_profile() {
    // GPT-OSS uses logits-based top-k routing with OpenAI SwiGLU experts.
    // Keep the profile conservative until GPT-OSS-specific PP shortcuts are measured.
    return qwen35_profile();
}

static const llama_moe_hot_cache_model_adapter ADAPTERS[] = {
    { LLM_ARCH_QWEN35MOE,  "qwen35moe", llama_moe_hot_cache_graph_kind::qwen35_ffn, LLM_FFN_SILU },
    { LLM_ARCH_QWEN3NEXT,  "qwen3next", llama_moe_hot_cache_graph_kind::logits,     LLM_FFN_SILU },
    { LLM_ARCH_GEMMA4,     "gemma4",    llama_moe_hot_cache_graph_kind::logits,     LLM_FFN_GELU },
    { LLM_ARCH_MELLUM,     "mellum",    llama_moe_hot_cache_graph_kind::logits,     LLM_FFN_SILU },
    { LLM_ARCH_OPENAI_MOE, "gpt-oss",   llama_moe_hot_cache_graph_kind::logits,     LLM_FFN_SWIGLU_OAI_MOE },
};

} // namespace

int llama_moe_hot_cache_graph_tweaks::parallel_mode() {
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

int64_t llama_moe_hot_cache_graph_tweaks::parallel_min_slots() {
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

bool llama_moe_hot_cache_graph_tweaks::merge_sum_rows() {
    static const bool enabled = env_enabled_by_default("LLAMA_MOE_HOT_CACHE_MERGE_SUM_ROWS");
    return enabled;
}

bool llama_moe_hot_cache_graph_tweaks::cpu_decode_routing() {
    static const bool enabled = env_enabled_by_default("LLAMA_MOE_HOT_CACHE_CPU_DECODE_ROUTING");
    return enabled;
}

bool llama_moe_hot_cache_graph_tweaks::decode_direct_merge() {
    static const bool enabled = env_enabled_by_default("LLAMA_MOE_HOT_CACHE_DECODE_DIRECT_MERGE");
    return enabled;
}

bool llama_moe_hot_cache_graph_tweaks::decode_strided_sum_rows() {
    static const bool enabled = env_enabled_by_default("LLAMA_MOE_HOT_CACHE_DECODE_STRIDED_SUM_ROWS");
    return enabled;
}

bool llama_moe_hot_cache_graph_tweaks::hot_dummy_padding() {
    static const bool enabled = env_enabled_by_default("LLAMA_MOE_HOT_CACHE_HOT_DUMMY_PADDING");
    return enabled;
}

bool llama_moe_hot_cache_graph_tweaks::shared_input_row() {
    static const bool enabled = env_enabled_by_default("LLAMA_MOE_HOT_CACHE_SHARED_INPUT_ROW");
    return enabled;
}

bool llama_moe_hot_cache_graph_tweaks::cold_prefix_sum() {
    static const bool enabled = env_enabled_by_default("LLAMA_MOE_HOT_CACHE_COLD_PREFIX_SUM");
    return enabled;
}

bool llama_moe_hot_cache_graph_tweaks::cold_prefix_weighted_sum() {
    static const bool enabled = env_enabled_by_default("LLAMA_MOE_HOT_CACHE_COLD_PREFIX_WEIGHTED_SUM");
    return enabled;
}

bool llama_moe_hot_cache_graph_tweaks::decode_repeat_hot_input() {
    static const bool enabled = env_enabled_by_default("LLAMA_MOE_HOT_CACHE_DECODE_REPEAT_HOT_INPUT");
    return enabled;
}

bool llama_moe_hot_cache_graph_tweaks::cold_first_row_input() {
    static const bool enabled = env_enabled_by_default("LLAMA_MOE_HOT_CACHE_COLD_FIRST_ROW_INPUT");
    return enabled;
}

bool llama_moe_hot_cache_graph_tweaks::branch_reduce_merge() {
    static const bool enabled = env_enabled_by_default("LLAMA_MOE_HOT_CACHE_BRANCH_REDUCE_MERGE");
    return enabled;
}

int llama_moe_hot_cache_graph_tweaks::prefix_reduce_tasks_max() {
    static const int value = []() {
        const char * env = std::getenv("LLAMA_MOE_HOT_CACHE_PREFIX_REDUCE_TASKS");
        if (env == nullptr || env[0] == '\0') {
            return 4;
        }

        char * end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end == env || parsed < 1) {
            return 4;
        }

        return (int) std::min<long>(parsed, 64);
    }();

    return value;
}

bool llama_moe_hot_cache_graph_tweaks::pp_reduce_merge(int64_t n_tokens, int64_t capacity) {
    return llama_moe_hot_cache_pp_policy::reduce_merge_enabled(n_tokens, capacity);
}

llama_moe_hot_cache_graph_profile llama_moe_hot_cache_model_adapter::profile() const {
    switch (arch) {
        case LLM_ARCH_QWEN35MOE:
            return qwen35_profile();
        case LLM_ARCH_QWEN3NEXT:
            return qwen3next_profile();
        case LLM_ARCH_GEMMA4:
            return gemma4_profile();
        case LLM_ARCH_MELLUM:
            return mellum_profile();
        case LLM_ARCH_OPENAI_MOE:
            return openai_moe_profile();
        default:
            return {};
    }
}

const llama_moe_hot_cache_model_adapter * llama_moe_hot_cache_find_model_adapter(llm_arch arch) {
    for (const auto & adapter : ADAPTERS) {
        if (adapter.arch == arch) {
            return &adapter;
        }
    }

    return nullptr;
}

const llama_moe_hot_cache_model_adapter * llama_moe_hot_cache_find_model_adapter(
        llm_arch arch,
        llama_moe_hot_cache_graph_kind graph_kind) {
    const llama_moe_hot_cache_model_adapter * adapter = llama_moe_hot_cache_find_model_adapter(arch);
    if (adapter == nullptr) {
        return nullptr;
    }

    if (graph_kind != llama_moe_hot_cache_graph_kind::none && adapter->graph_kind != graph_kind) {
        return nullptr;
    }

    return adapter;
}

bool llama_moe_hot_cache_adapter_supports_arch(llm_arch arch) {
    return llama_moe_hot_cache_find_model_adapter(arch) != nullptr;
}

bool llama_moe_hot_cache_adapter_supports_graph_kind(
        llm_arch arch,
        llama_moe_hot_cache_graph_kind graph_kind) {
    return llama_moe_hot_cache_find_model_adapter(arch, graph_kind) != nullptr;
}

const char * llama_moe_hot_cache_graph_kind_name(llama_moe_hot_cache_graph_kind graph_kind) {
    switch (graph_kind) {
        case llama_moe_hot_cache_graph_kind::qwen35_ffn:
            return "qwen35_ffn";
        case llama_moe_hot_cache_graph_kind::logits:
            return "logits";
        case llama_moe_hot_cache_graph_kind::none:
        default:
            return "none";
    }
}

llama_moe_hot_cache_graph_profile llama_moe_hot_cache_graph_profile_for_arch(llm_arch arch) {
    const llama_moe_hot_cache_model_adapter * adapter = llama_moe_hot_cache_find_model_adapter(arch);
    return adapter != nullptr ? adapter->profile() : llama_moe_hot_cache_graph_profile{};
}
