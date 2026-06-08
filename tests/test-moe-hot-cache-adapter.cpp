#include "../src/moe-hot-cache/llama-moe-hot-cache-adapter.h"
#include "../include/llama.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

static void require_impl(bool condition, int line) {
    if (!condition) {
        throw std::runtime_error("test assertion failed at line " + std::to_string(line));
    }
}

#define require(condition) require_impl((condition), __LINE__)

static void set_env_var(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
#endif
}

static void clear_profile_env() {
    set_env_var("LLAMA_MOE_HOT_CACHE_MERGE_SUM_ROWS", nullptr);
    set_env_var("LLAMA_MOE_HOT_CACHE_CPU_DECODE_ROUTING", nullptr);
    set_env_var("LLAMA_MOE_HOT_CACHE_DECODE_DIRECT_MERGE", nullptr);
    set_env_var("LLAMA_MOE_HOT_CACHE_DECODE_STRIDED_SUM_ROWS", nullptr);
    set_env_var("LLAMA_MOE_HOT_CACHE_HOT_DUMMY_PADDING", nullptr);
    set_env_var("LLAMA_MOE_HOT_CACHE_SHARED_INPUT_ROW", nullptr);
    set_env_var("LLAMA_MOE_HOT_CACHE_COLD_PREFIX_SUM", nullptr);
    set_env_var("LLAMA_MOE_HOT_CACHE_COLD_PREFIX_WEIGHTED_SUM", nullptr);
    set_env_var("LLAMA_MOE_HOT_CACHE_DECODE_REPEAT_HOT_INPUT", nullptr);
    set_env_var("LLAMA_MOE_HOT_CACHE_COLD_FIRST_ROW_INPUT", nullptr);
    set_env_var("LLAMA_MOE_HOT_CACHE_BRANCH_REDUCE_MERGE", nullptr);
    set_env_var("LLAMA_MOE_HOT_CACHE_PREFIX_REDUCE_TASKS", nullptr);
    set_env_var("LLAMA_MOE_HOT_CACHE_PP_REDUCE_MERGE", nullptr);
}

static void test_find_supported_adapters() {
    const auto * qwen = llama_moe_hot_cache_find_model_adapter(LLM_ARCH_QWEN35MOE);
    require(qwen != nullptr);
    require(std::string(qwen->name) == "qwen35moe");
    require(qwen->graph_kind == llama_moe_hot_cache_graph_kind::qwen35_ffn);
    require(qwen->ffn_op == LLM_FFN_SILU);

    const auto * qwen_next = llama_moe_hot_cache_find_model_adapter(LLM_ARCH_QWEN3NEXT);
    require(qwen_next != nullptr);
    require(std::string(qwen_next->name) == "qwen3next");
    require(qwen_next->graph_kind == llama_moe_hot_cache_graph_kind::logits);
    require(qwen_next->ffn_op == LLM_FFN_SILU);

    const auto * gemma = llama_moe_hot_cache_find_model_adapter(LLM_ARCH_GEMMA4);
    require(gemma != nullptr);
    require(std::string(gemma->name) == "gemma4");
    require(gemma->graph_kind == llama_moe_hot_cache_graph_kind::logits);
    require(gemma->ffn_op == LLM_FFN_GELU);

    const auto * mellum = llama_moe_hot_cache_find_model_adapter(LLM_ARCH_MELLUM);
    require(mellum != nullptr);
    require(std::string(mellum->name) == "mellum");
    require(mellum->graph_kind == llama_moe_hot_cache_graph_kind::logits);
    require(mellum->ffn_op == LLM_FFN_SILU);

    const auto * openai_moe = llama_moe_hot_cache_find_model_adapter(LLM_ARCH_OPENAI_MOE);
    require(openai_moe != nullptr);
    require(std::string(openai_moe->name) == "gpt-oss");
    require(openai_moe->graph_kind == llama_moe_hot_cache_graph_kind::logits);
    require(openai_moe->ffn_op == LLM_FFN_SWIGLU_OAI_MOE);
}

static void test_rejects_unsupported_arch() {
    require(!llama_moe_hot_cache_adapter_supports_arch(LLM_ARCH_LLAMA));
    require(llama_moe_hot_cache_find_model_adapter(LLM_ARCH_LLAMA) == nullptr);
    require(llama_moe_hot_cache_find_model_adapter(LLM_ARCH_LLAMA, llama_moe_hot_cache_graph_kind::logits) == nullptr);
    require(!llama_moe_hot_cache_adapter_supports_graph_kind(LLM_ARCH_LLAMA, llama_moe_hot_cache_graph_kind::logits));

    const auto profile = llama_moe_hot_cache_graph_profile_for_arch(LLM_ARCH_LLAMA);
    require(!profile.cpu_decode_routing);
    require(!profile.decode_direct_merge);
    require(!profile.merge_sum_rows);
    require(!profile.branch_reduce_merge);
}

static void test_graph_kind_capability_checks() {
    require(llama_moe_hot_cache_adapter_supports_graph_kind(LLM_ARCH_QWEN35MOE, llama_moe_hot_cache_graph_kind::qwen35_ffn));
    require(!llama_moe_hot_cache_adapter_supports_graph_kind(LLM_ARCH_QWEN35MOE, llama_moe_hot_cache_graph_kind::logits));

    require(llama_moe_hot_cache_adapter_supports_graph_kind(LLM_ARCH_GEMMA4, llama_moe_hot_cache_graph_kind::logits));
    require(!llama_moe_hot_cache_adapter_supports_graph_kind(LLM_ARCH_GEMMA4, llama_moe_hot_cache_graph_kind::qwen35_ffn));

    require(llama_moe_hot_cache_adapter_supports_graph_kind(LLM_ARCH_MELLUM, llama_moe_hot_cache_graph_kind::logits));
    require(!llama_moe_hot_cache_adapter_supports_graph_kind(LLM_ARCH_MELLUM, llama_moe_hot_cache_graph_kind::qwen35_ffn));

    require(llama_moe_hot_cache_adapter_supports_graph_kind(LLM_ARCH_OPENAI_MOE, llama_moe_hot_cache_graph_kind::logits));
    require(!llama_moe_hot_cache_adapter_supports_graph_kind(LLM_ARCH_OPENAI_MOE, llama_moe_hot_cache_graph_kind::qwen35_ffn));

    const auto * qwen_any = llama_moe_hot_cache_find_model_adapter(LLM_ARCH_QWEN35MOE, llama_moe_hot_cache_graph_kind::none);
    require(qwen_any != nullptr);
    require(qwen_any->graph_kind == llama_moe_hot_cache_graph_kind::qwen35_ffn);

    require(std::string(llama_moe_hot_cache_graph_kind_name(llama_moe_hot_cache_graph_kind::none)) == "none");
    require(std::string(llama_moe_hot_cache_graph_kind_name(llama_moe_hot_cache_graph_kind::qwen35_ffn)) == "qwen35_ffn");
    require(std::string(llama_moe_hot_cache_graph_kind_name(llama_moe_hot_cache_graph_kind::logits)) == "logits");
}

static void test_profile_defaults_are_arch_specific() {
    clear_profile_env();

    const auto qwen = llama_moe_hot_cache_graph_profile_for_arch(LLM_ARCH_QWEN35MOE);
    require(qwen.cpu_decode_routing);
    require(qwen.decode_direct_merge);
    require(qwen.merge_sum_rows);
    require(!qwen.branch_reduce_merge);
    require(qwen.cpu_decode_routing_max_tokens == 1);

    const auto qwen_next = llama_moe_hot_cache_graph_profile_for_arch(LLM_ARCH_QWEN3NEXT);
    require(qwen_next.cpu_decode_routing);
    require(qwen_next.cpu_decode_routing_max_tokens == 4);
    require(qwen_next.prefix_reduce_tasks_max == 1);

    const auto gemma = llama_moe_hot_cache_graph_profile_for_arch(LLM_ARCH_GEMMA4);
    require(gemma.cpu_decode_routing);
    require(gemma.decode_direct_merge);
    require(gemma.branch_reduce_merge);
    require(gemma.cpu_decode_routing_max_tokens == 1);

    const auto mellum = llama_moe_hot_cache_graph_profile_for_arch(LLM_ARCH_MELLUM);
    require(mellum.cpu_decode_routing);
    require(mellum.decode_direct_merge);
    require(mellum.merge_sum_rows);
    require(!mellum.branch_reduce_merge);
    require(mellum.cpu_decode_routing_max_tokens == 1);

    const auto openai_moe = llama_moe_hot_cache_graph_profile_for_arch(LLM_ARCH_OPENAI_MOE);
    require(openai_moe.cpu_decode_routing);
    require(openai_moe.decode_direct_merge);
    require(openai_moe.merge_sum_rows);
    require(!openai_moe.branch_reduce_merge);
    require(openai_moe.cpu_decode_routing_max_tokens == 1);
}

static void test_parallel_mode_is_runtime_switchable() {
    set_env_var("LLAMA_MOE_HOT_CACHE_PARALLEL", nullptr);
    require(llama_moe_hot_cache_graph_tweaks::parallel_mode() == 1);

    set_env_var("LLAMA_MOE_HOT_CACHE_PARALLEL", "0");
    require(llama_moe_hot_cache_graph_tweaks::parallel_mode() == 0);

    set_env_var("LLAMA_MOE_HOT_CACHE_PARALLEL", "force");
    require(llama_moe_hot_cache_graph_tweaks::parallel_mode() == 2);

    set_env_var("LLAMA_MOE_HOT_CACHE_PARALLEL", nullptr);
}

static void test_default_worker_lane_reserve() {
    const auto params = llama_model_default_params();
    require(params.moe_hot_cache_auto_reserve_mib == 1024);
    require(params.moe_hot_cache_second_auto_reserve_mib == 512);
    require(params.moe_hot_cache_third_auto_reserve_mib == 512);
}

int main() {
    test_find_supported_adapters();
    test_rejects_unsupported_arch();
    test_graph_kind_capability_checks();
    test_profile_defaults_are_arch_specific();
    test_parallel_mode_is_runtime_switchable();
    test_default_worker_lane_reserve();
    return 0;
}
