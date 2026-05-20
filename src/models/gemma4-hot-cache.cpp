#include "../llama-moe-hot-cache.h"

#include <cstdlib>

std::vector<llama_moe_hot_cache_entry> llama_moe_hot_cache_gemma4_weighting::score_observations(
        const std::vector<llama_moe_hot_cache_layer_observation> & observations) {
    auto config = llama_moe_hot_cache_weighting::default_config();
    const char * curve = std::getenv("LLAMA_MOE_HOT_CACHE_GEMMA4_LAYER_CURVE");
    if (curve != nullptr && curve[0] != '\0') {
        char * end = nullptr;
        const double parsed = std::strtod(curve, &end);
        if (end != curve) {
            config.layer_curve = parsed;
        }
    }

    return llama_moe_hot_cache_weighting::score_observations(observations, config);
}
