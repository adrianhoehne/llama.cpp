#include "llama-moe-hot-cache-parser.h"

#include "ggml.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

using json = nlohmann::ordered_json;

static void add_saturating(uint64_t & dst, uint64_t value) {
    if (dst > std::numeric_limits<uint64_t>::max() - value) {
        dst = std::numeric_limits<uint64_t>::max();
    } else {
        dst += value;
    }
}

static llama_moe_hot_cache_expert_observation & find_or_add_expert_observation(
        llama_moe_hot_cache_layer_observation & layer,
        uint32_t expert) {
    for (auto & existing : layer.experts) {
        if (existing.expert == expert) {
            return existing;
        }
    }

    layer.experts.push_back({ expert, 0, 0, 0 });
    return layer.experts.back();
}

} // namespace

std::vector<llama_moe_hot_cache_layer_observation> llama_moe_hot_cache_perf_json_parser::parse_observations(
        const std::string & json_str) {
    json root;
    try {
        root = json::parse(json_str);
    } catch (const std::exception & e) {
        throw std::runtime_error(std::string("failed to parse --moe-hot-cache JSON: ") + e.what());
    }

    if (!root.is_object()) {
        throw std::runtime_error("--moe-hot-cache JSON must use schema llama.cpp.moe_layer_perf.v1 or llama.cpp.moe_layer_opt_perf.v1");
    }

    const std::string schema = root.value("schema", "");
    if (schema != "llama.cpp.moe_layer_perf.v1" &&
        schema != "llama.cpp.moe_layer_opt_perf.v1") {
        throw std::runtime_error("--moe-hot-cache JSON must use schema llama.cpp.moe_layer_perf.v1 or llama.cpp.moe_layer_opt_perf.v1");
    }

    if (!root.contains("layers") || !root["layers"].is_array()) {
        throw std::runtime_error("--moe-hot-cache JSON must contain a layers array");
    }

    std::vector<llama_moe_hot_cache_layer_observation> observations;
    for (const auto & layer : root["layers"]) {
        if (!layer.is_object() || !layer.contains("layer") || !layer["layer"].is_number_unsigned()) {
            throw std::runtime_error("--moe-hot-cache layer entries must contain an unsigned layer id");
        }

        llama_moe_hot_cache_layer_observation obs;
        obs.layer = layer["layer"].get<uint32_t>();

        const auto add_expert_counts = [&](const char * field_name, uint64_t llama_moe_hot_cache_expert_observation::* member) {
            const auto it = layer.find(field_name);
            if (it == layer.end()) {
                return false;
            }
            if (!it->is_array()) {
                throw std::runtime_error(std::string("--moe-hot-cache ") + field_name + " must be an array");
            }

            bool added_any = false;
            for (const auto & expert : *it) {
                if (!expert.is_array() || expert.size() != 2 ||
                    !expert[0].is_number_unsigned() || !expert[1].is_number_unsigned()) {
                    throw std::runtime_error(std::string("--moe-hot-cache ") + field_name + " must be [expert_id, hit_count] arrays");
                }

                const uint64_t hit_count = expert[1].get<uint64_t>();
                if (hit_count == 0) {
                    continue;
                }

                const uint32_t expert_id = expert[0].get<uint32_t>();
                auto & counts = find_or_add_expert_observation(obs, expert_id);
                add_saturating(counts.*member, hit_count);
                added_any = true;
            }

            return added_any;
        };

        obs.has_branch_counts =
            add_expert_counts("hot_experts",  &llama_moe_hot_cache_expert_observation::hot) |
            add_expert_counts("cold_experts", &llama_moe_hot_cache_expert_observation::cold);

        if (!obs.has_branch_counts && !add_expert_counts("experts", &llama_moe_hot_cache_expert_observation::raw)) {
            continue;
        }

        obs.cold_slots_per_call = layer.value("cold_slots_per_call", 0.0);
        if (obs.cold_slots_per_call <= 0.0) {
            const uint64_t calls = layer.value("calls", uint64_t(0));
            const uint64_t cold_slots_total = layer.value("cold_slots_total", uint64_t(0));
            if (calls > 0 && cold_slots_total > 0) {
                obs.cold_slots_per_call = (double) cold_slots_total / (double) calls;
            }
        }

        obs.parallel_join_wait_time_per_call_us = layer.value("parallel_join_wait_time_per_call_us", 0.0);
        obs.parallel_cold_lane_wall_time_per_call_us = layer.value("parallel_cold_lane_wall_time_per_call_us", 0.0);
        obs.parallel_hot_lane_wall_time_per_call_us = layer.value("parallel_hot_lane_wall_time_per_call_us", 0.0);
        obs.total_moe_time_per_call_us = layer.value("total_moe_time_per_call_us", 0.0);

        double wait_per_call = obs.parallel_join_wait_time_per_call_us;
        if (wait_per_call <= 0.0) {
            wait_per_call = std::max(0.0, obs.parallel_cold_lane_wall_time_per_call_us - obs.parallel_hot_lane_wall_time_per_call_us);
        }

        if (wait_per_call > 0.0 && obs.cold_slots_per_call > 0.0) {
            obs.wait_per_cold_slot_us = wait_per_call / obs.cold_slots_per_call;
        }

        std::sort(obs.experts.begin(), obs.experts.end(), [](const auto & a, const auto & b) {
            return a.expert < b.expert;
        });

        if (!obs.experts.empty()) {
            observations.push_back(std::move(obs));
        }
    }

    std::sort(observations.begin(), observations.end(), [](const auto & a, const auto & b) {
        return a.layer < b.layer;
    });

    return observations;
}

bool llama_moe_hot_cache_perf_json_parser::parse_enabled_layer_slots(
        const std::string & json_str,
        std::vector<llama_moe_hot_cache_perf_json_layer_slots> & layer_slots) {
    layer_slots.clear();

    json root;
    root = json::parse(json_str);

    if (!root.is_object() || !root.value("enabled", false) ||
        !root.contains("layers") || !root["layers"].is_array()) {
        return false;
    }

    for (const auto & layer : root["layers"]) {
        if (!layer.is_object() || !layer.contains("layer") || !layer["layer"].is_number_unsigned()) {
            continue;
        }

        layer_slots.push_back({
            layer["layer"].get<uint32_t>(),
            layer.value("hot_slots_total", uint64_t(0)),
            layer.value("cold_slots_total", uint64_t(0)),
        });
    }

    return true;
}
