#pragma once

#include "llama-moe-hot-cache.h"

#include <string>
#include <vector>

struct llama_moe_hot_cache_perf_json_layer_slots {
    uint32_t layer = 0;
    uint64_t hot_slots = 0;
    uint64_t cold_slots = 0;
};

class llama_moe_hot_cache_perf_json_parser {
public:
    static std::vector<llama_moe_hot_cache_layer_observation> parse_observations(
            const std::string & json_str);

    static bool parse_enabled_layer_slots(
            const std::string & json_str,
            std::vector<llama_moe_hot_cache_perf_json_layer_slots> & layer_slots);
};
