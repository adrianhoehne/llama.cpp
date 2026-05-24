#include "../moe-hot-cache/llama-moe-hot-cache.h"

bool llama_moe_hot_cache_qwen35moe_weighting::parse_mode(
        const std::string & name,
        llama_moe_hot_cache_qwen35moe_weighting_mode & mode) {
    return llama_moe_hot_cache_weighting::parse_mode(name, mode);
}

const char * llama_moe_hot_cache_qwen35moe_weighting::mode_name(
        llama_moe_hot_cache_qwen35moe_weighting_mode mode) {
    return llama_moe_hot_cache_weighting::mode_name(mode);
}

llama_moe_hot_cache_qwen35moe_weighting_config llama_moe_hot_cache_qwen35moe_weighting::default_config() {
    return llama_moe_hot_cache_weighting::default_config();
}

std::vector<llama_moe_hot_cache_entry> llama_moe_hot_cache_qwen35moe_weighting::score_observations(
        const std::vector<llama_moe_hot_cache_layer_observation> & observations) {
    return llama_moe_hot_cache_weighting::score_observations(observations);
}

std::vector<llama_moe_hot_cache_entry> llama_moe_hot_cache_qwen35moe_weighting::score_observations(
        const std::vector<llama_moe_hot_cache_layer_observation> & observations,
        const llama_moe_hot_cache_qwen35moe_weighting_config & config) {
    return llama_moe_hot_cache_weighting::score_observations(observations, config);
}
