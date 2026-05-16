#include "../llama-moe-hot-cache.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>

namespace {

static constexpr double QWEN35MOE_LAYER_CURVE_DEFAULT = 0.50;
static constexpr double QWEN35MOE_HOT_STICKY_BONUS = 0.05;

static double qwen35moe_layer_curve() {
    static const double value = []() {
        const char * env = std::getenv("LLAMA_MOE_HOT_CACHE_QWEN_LAYER_CURVE");
        if (env == nullptr || env[0] == '\0') {
            return QWEN35MOE_LAYER_CURVE_DEFAULT;
        }

        char * end = nullptr;
        const double parsed = std::strtod(env, &end);
        if (end == env || parsed < 0.0 || parsed > 1.0) {
            return QWEN35MOE_LAYER_CURVE_DEFAULT;
        }

        return parsed;
    }();

    return value;
}

static void add_saturating(uint64_t & dst, uint64_t value) {
    if (dst > std::numeric_limits<uint64_t>::max() - value) {
        dst = std::numeric_limits<uint64_t>::max();
    } else {
        dst += value;
    }
}

static uint64_t total_hits(
        const llama_moe_hot_cache_layer_observation & layer,
        const llama_moe_hot_cache_expert_observation & expert) {
    uint64_t result = expert.raw;
    if (layer.has_branch_counts) {
        result = expert.hot;
        add_saturating(result, expert.cold);
    }

    return result;
}

static uint64_t score_to_u64(long double score) {
    if (score <= 0.0L) {
        return 0;
    }
    if (score >= (long double) std::numeric_limits<uint64_t>::max()) {
        return std::numeric_limits<uint64_t>::max();
    }

    return std::max<uint64_t>(1, (uint64_t) (score + 0.5L));
}

static double layer_pressure(const llama_moe_hot_cache_layer_observation & layer) {
    if (layer.parallel_join_wait_time_per_call_us > 0.0) {
        return layer.parallel_join_wait_time_per_call_us;
    }

    const double lane_delta =
        layer.parallel_cold_lane_wall_time_per_call_us -
        layer.parallel_hot_lane_wall_time_per_call_us;
    if (lane_delta > 0.0) {
        return lane_delta;
    }

    if (layer.cold_slots_per_call > 0.0) {
        return layer.cold_slots_per_call;
    }

    return layer.wait_per_cold_slot_us;
}

static double average_layer_pressure(const std::vector<llama_moe_hot_cache_layer_observation> & observations) {
    double sum = 0.0;
    int count = 0;

    for (const auto & layer : observations) {
        const double pressure = layer_pressure(layer);
        if (pressure > 0.0) {
            sum += pressure;
            ++count;
        }
    }

    return count > 0 ? sum / double(count) : 0.0;
}

static double layer_weight(const llama_moe_hot_cache_layer_observation & layer, double avg_pressure) {
    const double curve = qwen35moe_layer_curve();
    if (curve <= 0.0) {
        return 1.0;
    }

    const double pressure = layer_pressure(layer);
    if (avg_pressure <= 0.0 || pressure <= 0.0) {
        return 1.0;
    }

    const double damping = curve;
    const double min_weight = 1.0 - 0.30*curve;
    const double max_weight = 1.0 + 0.60*curve;
    const double relative = pressure / avg_pressure;
    const double weighted = 1.0 + damping*(relative - 1.0);
    return std::clamp(weighted, min_weight, max_weight);
}

static void sort_entries(std::vector<llama_moe_hot_cache_entry> & entries) {
    std::sort(entries.begin(), entries.end(), [](const auto & a, const auto & b) {
        if (a.hit_count != b.hit_count) {
            return a.hit_count > b.hit_count;
        }
        if (a.layer != b.layer) {
            return a.layer < b.layer;
        }
        return a.expert < b.expert;
    });
}

} // namespace

std::vector<llama_moe_hot_cache_entry> llama_moe_hot_cache_qwen35moe_weighting::score_observations(
        const std::vector<llama_moe_hot_cache_layer_observation> & observations) {
    const double avg_pressure = average_layer_pressure(observations);

    std::vector<llama_moe_hot_cache_entry> result;
    for (const auto & layer : observations) {
        const double weight = layer_weight(layer, avg_pressure);

        for (const auto & expert : layer.experts) {
            const uint64_t hits = total_hits(layer, expert);
            if (hits == 0) {
                continue;
            }

            long double score = (long double) hits * (long double) weight;
            if (layer.has_branch_counts && expert.hot > 0) {
                score += (long double) expert.hot * (long double) QWEN35MOE_HOT_STICKY_BONUS;
            }

            const uint64_t rounded = score_to_u64(score);
            if (rounded > 0) {
                result.push_back({ layer.layer, expert.expert, rounded });
            }
        }
    }

    sort_entries(result);
    return result;
}
