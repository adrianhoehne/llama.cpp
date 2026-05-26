#include "../src/moe-hot-cache/llama-moe-hot-cache-pp.h"

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

static void clear_env() {
    set_env_var("LLAMA_MOE_HOT_CACHE_PP_REDUCE_MERGE", nullptr);
    set_env_var("LLAMA_MOE_HOT_CACHE_PP_WORKLIST_ORDER", nullptr);
    set_env_var("LLAMA_MOE_HOT_CACHE_PP_COMPACT_COLD_REDUCE", nullptr);
    set_env_var("LLAMA_MOE_HOT_CACHE_PP_BYPASS", nullptr);
    set_env_var("LLAMA_MOE_HOT_CACHE_PP_BYPASS_MIN_TOKENS", nullptr);
    set_env_var("LLAMA_MOE_HOT_CACHE_PP_MIN_HOT_EXPERT_RATIO", nullptr);
}

static llama_moe_hot_cache_graph_profile profile_with_decode_branch_reduce() {
    llama_moe_hot_cache_graph_profile profile;
    profile.branch_reduce_merge = true;
    return profile;
}

static void test_phase_and_default_worklist_order() {
    clear_env();

    const auto warmup = llama_moe_hot_cache_pp_policy::build(
            llama_moe_hot_cache_graph_phase::warmup, 512, 4096, true, 8, {});
    require(warmup.phase == llama_moe_hot_cache_graph_phase::warmup);
    require(warmup.worklist_order == llama_moe_hot_cache_worklist_order::token_major);
    require(!warmup.branch_reduce_merge);
    require(std::string(llama_moe_hot_cache_graph_phase_name(warmup.phase)) == "warmup");

    const auto decode = llama_moe_hot_cache_pp_policy::build(
            llama_moe_hot_cache_graph_phase::decode, 1, 8, true, 8, profile_with_decode_branch_reduce());
    require(decode.phase == llama_moe_hot_cache_graph_phase::decode);
    require(decode.worklist_order == llama_moe_hot_cache_worklist_order::token_major);
    require(decode.branch_reduce_merge);
    require(std::string(llama_moe_hot_cache_graph_phase_name(decode.phase)) == "decode");

    const auto pp = llama_moe_hot_cache_pp_policy::build(
            llama_moe_hot_cache_graph_phase::prompt_processing, 128, 1024, true, 8, {});
    require(pp.phase == llama_moe_hot_cache_graph_phase::prompt_processing);
    require(pp.worklist_order == llama_moe_hot_cache_worklist_order::expert_major);
    require(!pp.branch_reduce_merge);
    require(!pp.compact_cold_reduce);
    require(pp.is_prompt_processing());
    require(std::string(llama_moe_hot_cache_graph_phase_name(pp.phase)) == "prompt_processing");

    const auto multi_token_decode = llama_moe_hot_cache_pp_policy::build(
            llama_moe_hot_cache_graph_phase::decode, 4, 40, true, 10, profile_with_decode_branch_reduce());
    require(multi_token_decode.phase == llama_moe_hot_cache_graph_phase::decode);
    require(multi_token_decode.worklist_order == llama_moe_hot_cache_worklist_order::token_major);
    require(!multi_token_decode.branch_reduce_merge);

    const auto single_token_pp_tail = llama_moe_hot_cache_pp_policy::build(
            llama_moe_hot_cache_graph_phase::prompt_processing, 1, 8, true, 8, profile_with_decode_branch_reduce());
    require(single_token_pp_tail.phase == llama_moe_hot_cache_graph_phase::prompt_processing);
    require(single_token_pp_tail.worklist_order == llama_moe_hot_cache_worklist_order::expert_major);
    require(!single_token_pp_tail.branch_reduce_merge);
}

static void test_reduce_merge_modes() {
    clear_env();

    set_env_var("LLAMA_MOE_HOT_CACHE_PP_REDUCE_MERGE", "on");
    require(llama_moe_hot_cache_pp_policy::reduce_merge_enabled(2, 16));
    require(llama_moe_hot_cache_pp_policy::build(
            llama_moe_hot_cache_graph_phase::prompt_processing, 2, 16, true, 8, {}).branch_reduce_merge);

    set_env_var("LLAMA_MOE_HOT_CACHE_PP_REDUCE_MERGE", "auto");
    require(!llama_moe_hot_cache_pp_policy::reduce_merge_enabled(16, 128));
    require(llama_moe_hot_cache_pp_policy::reduce_merge_enabled(32, 256));
    require(!llama_moe_hot_cache_pp_policy::build(
            llama_moe_hot_cache_graph_phase::prompt_processing, 32, 256, false, 8, {}).branch_reduce_merge);
    require(!llama_moe_hot_cache_pp_policy::build(
            llama_moe_hot_cache_graph_phase::prompt_processing, 32, 256, true, 1, {}).branch_reduce_merge);
    require(llama_moe_hot_cache_pp_policy::build(
            llama_moe_hot_cache_graph_phase::prompt_processing, 32, 256, true, 8, {}).branch_reduce_merge);

    set_env_var("LLAMA_MOE_HOT_CACHE_PP_REDUCE_MERGE", "off");
    require(!llama_moe_hot_cache_pp_policy::reduce_merge_enabled(128, 1024));
}

static void test_compact_cold_reduce_modes() {
    clear_env();

    require(!llama_moe_hot_cache_pp_policy::compact_cold_reduce_enabled(
            llama_moe_hot_cache_graph_phase::warmup, 128));
    require(!llama_moe_hot_cache_pp_policy::compact_cold_reduce_enabled(
            llama_moe_hot_cache_graph_phase::prompt_processing, 1));
    require(llama_moe_hot_cache_pp_policy::compact_cold_reduce_enabled(
            llama_moe_hot_cache_graph_phase::prompt_processing, 128));

    set_env_var("LLAMA_MOE_HOT_CACHE_PP_REDUCE_MERGE", "on");
    auto plan = llama_moe_hot_cache_pp_policy::build(
            llama_moe_hot_cache_graph_phase::prompt_processing, 128, 1024, true, 8, {});
    require(plan.branch_reduce_merge);
    require(plan.compact_cold_reduce);

    set_env_var("LLAMA_MOE_HOT_CACHE_PP_COMPACT_COLD_REDUCE", "off");
    plan = llama_moe_hot_cache_pp_policy::build(
            llama_moe_hot_cache_graph_phase::prompt_processing, 128, 1024, true, 8, {});
    require(plan.branch_reduce_merge);
    require(!plan.compact_cold_reduce);
}

static void test_worklist_order_modes() {
    clear_env();

    require(llama_moe_hot_cache_pp_policy::worklist_order(
            llama_moe_hot_cache_graph_phase::warmup) == llama_moe_hot_cache_worklist_order::token_major);
    require(llama_moe_hot_cache_pp_policy::worklist_order(
            llama_moe_hot_cache_graph_phase::decode) == llama_moe_hot_cache_worklist_order::token_major);
    require(llama_moe_hot_cache_pp_policy::worklist_order(
            llama_moe_hot_cache_graph_phase::prompt_processing) == llama_moe_hot_cache_worklist_order::expert_major);

    set_env_var("LLAMA_MOE_HOT_CACHE_PP_WORKLIST_ORDER", "token");
    require(llama_moe_hot_cache_pp_policy::worklist_order(
            llama_moe_hot_cache_graph_phase::prompt_processing) == llama_moe_hot_cache_worklist_order::token_major);

    set_env_var("LLAMA_MOE_HOT_CACHE_PP_WORKLIST_ORDER", "expert-major");
    require(llama_moe_hot_cache_pp_policy::worklist_order(
            llama_moe_hot_cache_graph_phase::prompt_processing) == llama_moe_hot_cache_worklist_order::expert_major);

    require(std::string(llama_moe_hot_cache_worklist_order_name(llama_moe_hot_cache_worklist_order::token_major)) == "token_major");
    require(std::string(llama_moe_hot_cache_worklist_order_name(llama_moe_hot_cache_worklist_order::expert_major)) == "expert_major");
}

static void test_bypass_hot_cache_for_prompt_processing() {
    clear_env();

    require(!llama_moe_hot_cache_pp_policy::bypass_hot_cache_for_prompt_processing(
            LLM_GRAPH_PHASE_PROMPT_PROCESSING, false, 128, 0));
    require(llama_moe_hot_cache_pp_policy::bypass_hot_cache_for_prompt_processing(
            LLM_GRAPH_PHASE_PROMPT_PROCESSING, false, 128, 1));
    require(!llama_moe_hot_cache_pp_policy::bypass_hot_cache_for_prompt_processing(
            LLM_GRAPH_PHASE_DECODE, false, 4, 1));
    require(!llama_moe_hot_cache_pp_policy::bypass_hot_cache_for_prompt_processing(
            LLM_GRAPH_PHASE_PROMPT_PROCESSING, true, 128, 1));
    require(!llama_moe_hot_cache_pp_policy::bypass_hot_cache_for_prompt_processing(
            LLM_GRAPH_PHASE_UNKNOWN, false, 1, 1));
    require(llama_moe_hot_cache_pp_policy::bypass_hot_cache_for_prompt_processing(
            LLM_GRAPH_PHASE_UNKNOWN, false, 2, 1));

    set_env_var("LLAMA_MOE_HOT_CACHE_PP_BYPASS_MIN_TOKENS", "512");
    require(!llama_moe_hot_cache_pp_policy::bypass_hot_cache_for_prompt_processing(
            LLM_GRAPH_PHASE_PROMPT_PROCESSING, false, 128, 1));
    require(llama_moe_hot_cache_pp_policy::bypass_hot_cache_for_prompt_processing(
            LLM_GRAPH_PHASE_PROMPT_PROCESSING, false, 512, 1));

    set_env_var("LLAMA_MOE_HOT_CACHE_PP_BYPASS", "off");
    require(!llama_moe_hot_cache_pp_policy::bypass_hot_cache_for_prompt_processing(
            LLM_GRAPH_PHASE_PROMPT_PROCESSING, false, 128, 1));

    set_env_var("LLAMA_MOE_HOT_CACHE_PP_BYPASS", "on");
    require(llama_moe_hot_cache_pp_policy::bypass_hot_cache_for_prompt_processing(
            LLM_GRAPH_PHASE_PROMPT_PROCESSING, false, 128, 0));
}

static void test_bypass_hot_cache_for_prompt_processing_hot_expert_ratio() {
    clear_env();

    require(!llama_moe_hot_cache_pp_policy::bypass_hot_cache_for_prompt_processing(
            LLM_GRAPH_PHASE_PROMPT_PROCESSING, false, 512, 0, 17, 256, 0.0));
    require(llama_moe_hot_cache_pp_policy::bypass_hot_cache_for_prompt_processing(
            LLM_GRAPH_PHASE_PROMPT_PROCESSING, false, 512, 0, 17, 256, 0.07));
    require(!llama_moe_hot_cache_pp_policy::bypass_hot_cache_for_prompt_processing(
            LLM_GRAPH_PHASE_PROMPT_PROCESSING, false, 512, 0, 18, 256, 0.07));
    require(!llama_moe_hot_cache_pp_policy::bypass_hot_cache_for_prompt_processing(
            LLM_GRAPH_PHASE_DECODE, false, 1, 0, 1, 256, 1.0));
    require(!llama_moe_hot_cache_pp_policy::bypass_hot_cache_for_prompt_processing(
            LLM_GRAPH_PHASE_PROMPT_PROCESSING, true, 512, 0, 1, 256, 1.0));

    set_env_var("LLAMA_MOE_HOT_CACHE_PP_MIN_HOT_EXPERT_RATIO", "0.05");
    require(llama_moe_hot_cache_pp_policy::bypass_hot_cache_for_prompt_processing(
            LLM_GRAPH_PHASE_PROMPT_PROCESSING, false, 512, 0, 12, 256, 0.0));
    require(!llama_moe_hot_cache_pp_policy::bypass_hot_cache_for_prompt_processing(
            LLM_GRAPH_PHASE_PROMPT_PROCESSING, false, 512, 0, 13, 256, 0.0));

    set_env_var("LLAMA_MOE_HOT_CACHE_PP_BYPASS", "off");
    require(!llama_moe_hot_cache_pp_policy::bypass_hot_cache_for_prompt_processing(
            LLM_GRAPH_PHASE_PROMPT_PROCESSING, false, 512, 0, 1, 256, 1.0));

    set_env_var("LLAMA_MOE_HOT_CACHE_PP_BYPASS", "on");
    require(llama_moe_hot_cache_pp_policy::bypass_hot_cache_for_prompt_processing(
            LLM_GRAPH_PHASE_PROMPT_PROCESSING, false, 512, 0, 256, 256, 0.0));
}

int main() {
    test_phase_and_default_worklist_order();
    test_reduce_merge_modes();
    test_compact_cold_reduce_modes();
    test_worklist_order_modes();
    test_bypass_hot_cache_for_prompt_processing();
    test_bypass_hot_cache_for_prompt_processing_hot_expert_ratio();
    clear_env();
    return 0;
}
