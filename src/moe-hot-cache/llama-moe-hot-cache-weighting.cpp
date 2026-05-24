#include "llama-moe-hot-cache.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

static constexpr double HOT_CACHE_LAYER_CURVE_DEFAULT = 0.50;
static constexpr double HOT_CACHE_HOT_STICKY_BONUS = 0.05;

enum class llama_moe_hot_cache_pressure_source {
    parallel_pressure,
    total_moe_time,
};

struct llama_moe_hot_cache_pressure_stats {
    double avg = 0.0;
    double min = 0.0;
    double max = 0.0;
};

static bool str_is(const char * actual, const char * expected) {
    return actual != nullptr && std::strcmp(actual, expected) == 0;
}

static bool str_is_any(const char * actual, const char * a, const char * b, const char * c = nullptr) {
    return str_is(actual, a) || str_is(actual, b) || (c != nullptr && str_is(actual, c));
}

static double normalize_layer_curve(double value) {
    if (value < 0.0 || value > 1.0 || !std::isfinite(value)) {
        return HOT_CACHE_LAYER_CURVE_DEFAULT;
    }

    return value;
}

static double layer_curve_from_env() {
    const char * env = std::getenv("LLAMA_MOE_HOT_CACHE_LAYER_CURVE");
    if (env == nullptr || env[0] == '\0') {
        env = std::getenv("LLAMA_MOE_HOT_CACHE_QWEN_LAYER_CURVE");
    }
    if (env == nullptr || env[0] == '\0') {
        env = std::getenv("LLAMA_MOE_HOT_CACHE_GEMMA4_LAYER_CURVE");
    }
    if (env == nullptr || env[0] == '\0') {
        return HOT_CACHE_LAYER_CURVE_DEFAULT;
    }

    char * end = nullptr;
    const double parsed = std::strtod(env, &end);
    if (end == env) {
        return HOT_CACHE_LAYER_CURVE_DEFAULT;
    }

    return normalize_layer_curve(parsed);
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

static double layer_pressure_for_source(
        const llama_moe_hot_cache_layer_observation & layer,
        llama_moe_hot_cache_pressure_source source) {
    if (source == llama_moe_hot_cache_pressure_source::total_moe_time && layer.total_moe_time_per_call_us > 0.0) {
        return layer.total_moe_time_per_call_us;
    }

    return layer_pressure(layer);
}

static double average_layer_pressure(
        const std::vector<llama_moe_hot_cache_layer_observation> & observations,
        llama_moe_hot_cache_pressure_source source) {
    double sum = 0.0;
    int count = 0;

    for (const auto & layer : observations) {
        const double pressure = layer_pressure_for_source(layer, source);
        if (pressure > 0.0) {
            sum += pressure;
            ++count;
        }
    }

    return count > 0 ? sum / double(count) : 0.0;
}

static void minmax_layer_pressure(
        const std::vector<llama_moe_hot_cache_layer_observation> & observations,
        llama_moe_hot_cache_pressure_source source,
        double & min_pressure,
        double & max_pressure) {
    min_pressure = std::numeric_limits<double>::max();
    max_pressure = 0.0;

    for (const auto & layer : observations) {
        const double pressure = layer_pressure_for_source(layer, source);
        if (pressure <= 0.0) {
            continue;
        }

        min_pressure = std::min(min_pressure, pressure);
        max_pressure = std::max(max_pressure, pressure);
    }

    if (min_pressure == std::numeric_limits<double>::max()) {
        min_pressure = 0.0;
    }
}

static double percentile_sorted(const std::vector<double> & values, double percentile) {
    if (values.empty()) {
        return 0.0;
    }

    const double clamped = std::clamp(percentile, 0.0, 1.0);
    const double pos = clamped*(double) (values.size() - 1);
    const size_t lo = (size_t) std::floor(pos);
    const size_t hi = (size_t) std::ceil(pos);
    if (lo == hi) {
        return values[lo];
    }

    const double t = pos - (double) lo;
    return values[lo]*(1.0 - t) + values[hi]*t;
}

static void robust_layer_pressure_bounds(
        const std::vector<llama_moe_hot_cache_layer_observation> & observations,
        llama_moe_hot_cache_pressure_source source,
        double & low_pressure,
        double & high_pressure) {
    std::vector<double> values;
    values.reserve(observations.size());

    for (const auto & layer : observations) {
        const double pressure = layer_pressure_for_source(layer, source);
        if (pressure > 0.0) {
            values.push_back(pressure);
        }
    }

    if (values.empty()) {
        low_pressure = 0.0;
        high_pressure = 0.0;
        return;
    }

    std::sort(values.begin(), values.end());
    low_pressure = percentile_sorted(values, 0.10);
    high_pressure = percentile_sorted(values, 0.90);

    if (high_pressure <= low_pressure) {
        low_pressure = values.front();
        high_pressure = values.back();
    }
}

static llama_moe_hot_cache_pressure_stats pressure_stats(
        const std::vector<llama_moe_hot_cache_layer_observation> & observations,
        llama_moe_hot_cache_pressure_source source,
        bool robust_bounds) {
    llama_moe_hot_cache_pressure_stats stats;
    stats.avg = average_layer_pressure(observations, source);

    if (robust_bounds) {
        robust_layer_pressure_bounds(observations, source, stats.min, stats.max);
    } else {
        minmax_layer_pressure(observations, source, stats.min, stats.max);
    }

    return stats;
}

static double pressure_weight(
        const llama_moe_hot_cache_layer_observation & layer,
        double curve,
        const llama_moe_hot_cache_pressure_stats & stats) {
    if (curve <= 0.0) {
        return 1.0;
    }

    const double pressure = layer_pressure_for_source(layer, llama_moe_hot_cache_pressure_source::parallel_pressure);
    if (stats.avg <= 0.0 || pressure <= 0.0) {
        return 1.0;
    }

    const double damping = curve;
    const double min_weight = 1.0 - 0.30*curve;
    const double max_weight = 1.0 + 0.60*curve;
    const double relative = pressure / stats.avg;
    const double weighted = 1.0 + damping*(relative - 1.0);
    return std::clamp(weighted, min_weight, max_weight);
}

static double time_weight(
        const llama_moe_hot_cache_layer_observation & layer,
        double curve,
        const llama_moe_hot_cache_pressure_stats & stats) {
    if (curve <= 0.0) {
        return 1.0;
    }

    const double pressure = layer_pressure_for_source(layer, llama_moe_hot_cache_pressure_source::total_moe_time);
    if (stats.avg <= 0.0 || pressure <= 0.0 || stats.max <= stats.min) {
        return 1.0;
    }

    const double normalized = (pressure - stats.min) / (stats.max - stats.min);
    const double min_weight = 1.0 - 0.60*curve;
    const double max_weight = 1.0 + 0.80*curve;
    return std::clamp(min_weight + normalized*(max_weight - min_weight), min_weight, max_weight);
}

static double smooth_pressure_weight(
        const llama_moe_hot_cache_layer_observation & layer,
        double curve,
        const llama_moe_hot_cache_pressure_stats & stats) {
    if (curve <= 0.0) {
        return 1.0;
    }

    const double pressure = layer_pressure_for_source(layer, llama_moe_hot_cache_pressure_source::parallel_pressure);
    if (stats.avg <= 0.0 || pressure <= 0.0 || stats.max <= stats.min) {
        return 1.0;
    }

    const double normalized = std::clamp((pressure - stats.min) / (stats.max - stats.min), 0.0, 1.0);
    const double shaped = std::sqrt(normalized);
    const double max_boost = 0.30*curve;
    return 1.0 + max_boost*shaped;
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

template <typename WeightFn>
static std::vector<llama_moe_hot_cache_entry> score_with_layer_weight(
        const std::vector<llama_moe_hot_cache_layer_observation> & observations,
        WeightFn weight_fn) {
    std::vector<llama_moe_hot_cache_entry> result;
    for (const auto & layer : observations) {
        const double weight = weight_fn(layer);

        for (const auto & expert : layer.experts) {
            const uint64_t hits = total_hits(layer, expert);
            if (hits == 0) {
                continue;
            }

            long double score = (long double) hits * (long double) weight;
            if (layer.has_branch_counts && expert.hot > 0) {
                score += (long double) expert.hot * (long double) HOT_CACHE_HOT_STICKY_BONUS;
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

class llama_moe_hot_cache_weighting_strategy {
public:
    virtual ~llama_moe_hot_cache_weighting_strategy() = default;

    virtual std::vector<llama_moe_hot_cache_entry> score(
            const std::vector<llama_moe_hot_cache_layer_observation> & observations,
            double layer_curve) const = 0;
};

class llama_moe_hot_cache_pressure_weighting : public llama_moe_hot_cache_weighting_strategy {
public:
    std::vector<llama_moe_hot_cache_entry> score(
            const std::vector<llama_moe_hot_cache_layer_observation> & observations,
            double layer_curve) const override {
        const auto stats = pressure_stats(observations, llama_moe_hot_cache_pressure_source::parallel_pressure, false);
        return score_with_layer_weight(observations, [&](const auto & layer) {
            return pressure_weight(layer, layer_curve, stats);
        });
    }
};

class llama_moe_hot_cache_smooth_pressure_weighting : public llama_moe_hot_cache_weighting_strategy {
public:
    std::vector<llama_moe_hot_cache_entry> score(
            const std::vector<llama_moe_hot_cache_layer_observation> & observations,
            double layer_curve) const override {
        const auto stats = pressure_stats(observations, llama_moe_hot_cache_pressure_source::parallel_pressure, true);
        return score_with_layer_weight(observations, [&](const auto & layer) {
            return smooth_pressure_weight(layer, layer_curve, stats);
        });
    }
};

class llama_moe_hot_cache_time_weighting : public llama_moe_hot_cache_weighting_strategy {
public:
    std::vector<llama_moe_hot_cache_entry> score(
            const std::vector<llama_moe_hot_cache_layer_observation> & observations,
            double layer_curve) const override {
        const auto stats = pressure_stats(observations, llama_moe_hot_cache_pressure_source::total_moe_time, false);
        return score_with_layer_weight(observations, [&](const auto & layer) {
            return time_weight(layer, layer_curve, stats);
        });
    }
};

class llama_moe_hot_cache_balanced_weighting : public llama_moe_hot_cache_weighting_strategy {
public:
    std::vector<llama_moe_hot_cache_entry> score(
            const std::vector<llama_moe_hot_cache_layer_observation> & observations,
            double layer_curve) const override {
        GGML_UNUSED(layer_curve);

        std::vector<llama_moe_hot_cache_entry> result;
        for (const auto & layer : observations) {
            std::vector<llama_moe_hot_cache_entry> ranked;
            ranked.reserve(layer.experts.size());

            for (const auto & expert : layer.experts) {
                const uint64_t hits = total_hits(layer, expert);
                if (hits > 0) {
                    ranked.push_back({ layer.layer, expert.expert, hits });
                }
            }

            sort_entries(ranked);
            for (size_t i = 0; i < ranked.size(); ++i) {
                const uint64_t rank_score = (uint64_t) (ranked.size() - i);
                ranked[i].hit_count = rank_score*1000000ULL + std::min<uint64_t>(ranked[i].hit_count, 999999ULL);
                result.push_back(ranked[i]);
            }
        }

        sort_entries(result);
        return result;
    }
};

class llama_moe_hot_cache_flat_weighting : public llama_moe_hot_cache_weighting_strategy {
public:
    std::vector<llama_moe_hot_cache_entry> score(
            const std::vector<llama_moe_hot_cache_layer_observation> & observations,
            double layer_curve) const override {
        GGML_UNUSED(layer_curve);

        std::vector<std::vector<llama_moe_hot_cache_entry>> ranked_by_layer;
        ranked_by_layer.reserve(observations.size());

        size_t max_rank = 0;
        size_t total_entries = 0;
        for (const auto & layer : observations) {
            std::vector<llama_moe_hot_cache_entry> ranked;
            ranked.reserve(layer.experts.size());

            for (const auto & expert : layer.experts) {
                const uint64_t hits = total_hits(layer, expert);
                if (hits > 0) {
                    ranked.push_back({ layer.layer, expert.expert, hits });
                }
            }

            sort_entries(ranked);
            max_rank = std::max(max_rank, ranked.size());
            total_entries += ranked.size();
            ranked_by_layer.push_back(std::move(ranked));
        }

        std::vector<llama_moe_hot_cache_entry> result;
        result.reserve(total_entries);

        for (size_t rank = 0; rank < max_rank; ++rank) {
            for (const auto & ranked : ranked_by_layer) {
                if (rank >= ranked.size()) {
                    continue;
                }

                llama_moe_hot_cache_entry entry = ranked[rank];
                const uint64_t rank_score = (uint64_t) (max_rank - rank);
                entry.hit_count = rank_score*1000000ULL + std::min<uint64_t>(entry.hit_count, 999999ULL);
                result.push_back(entry);
            }
        }

        sort_entries(result);
        return result;
    }
};

static const llama_moe_hot_cache_weighting_strategy & weighting_strategy(llama_moe_hot_cache_weighting_mode mode) {
    static const llama_moe_hot_cache_pressure_weighting pressure;
    static const llama_moe_hot_cache_smooth_pressure_weighting smooth_pressure;
    static const llama_moe_hot_cache_time_weighting time;
    static const llama_moe_hot_cache_balanced_weighting balanced;
    static const llama_moe_hot_cache_flat_weighting flat;

    switch (mode) {
        case llama_moe_hot_cache_weighting_mode::smooth_pressure:
            return smooth_pressure;
        case llama_moe_hot_cache_weighting_mode::time:
            return time;
        case llama_moe_hot_cache_weighting_mode::balanced:
            return balanced;
        case llama_moe_hot_cache_weighting_mode::flat:
            return flat;
        case llama_moe_hot_cache_weighting_mode::pressure:
        default:
            return pressure;
    }
}

} // namespace

bool llama_moe_hot_cache_weighting::parse_mode(
        const std::string & name,
        llama_moe_hot_cache_weighting_mode & mode) {
    const char * value = name.c_str();
    if (str_is(value, "pressure")) {
        mode = llama_moe_hot_cache_weighting_mode::pressure;
        return true;
    }

    if (str_is_any(value, "smooth", "smooth-pressure") ||
        str_is_any(value, "capped", "capped-pressure", "soft-pressure")) {
        mode = llama_moe_hot_cache_weighting_mode::smooth_pressure;
        return true;
    }

    if (str_is_any(value, "time", "moe-time", "decode-time")) {
        mode = llama_moe_hot_cache_weighting_mode::time;
        return true;
    }

    if (str_is_any(value, "balanced", "rank", "layer-rank")) {
        mode = llama_moe_hot_cache_weighting_mode::balanced;
        return true;
    }

    if (str_is(value, "flat")) {
        mode = llama_moe_hot_cache_weighting_mode::flat;
        return true;
    }

    return false;
}

const char * llama_moe_hot_cache_weighting::mode_name(
        llama_moe_hot_cache_weighting_mode mode) {
    switch (mode) {
        case llama_moe_hot_cache_weighting_mode::smooth_pressure:
            return "smooth";
        case llama_moe_hot_cache_weighting_mode::time:
            return "time";
        case llama_moe_hot_cache_weighting_mode::balanced:
            return "balanced";
        case llama_moe_hot_cache_weighting_mode::flat:
            return "flat";
        case llama_moe_hot_cache_weighting_mode::pressure:
        default:
            return "pressure";
    }
}

llama_moe_hot_cache_weighting_config llama_moe_hot_cache_weighting::default_config() {
    llama_moe_hot_cache_weighting_config config;
    config.layer_curve = layer_curve_from_env();

    const char * env = std::getenv("LLAMA_MOE_HOT_CACHE_WEIGHTING");
    if (env == nullptr || env[0] == '\0') {
        env = std::getenv("LLAMA_MOE_HOT_CACHE_QWEN_WEIGHTING");
    }
    if (env != nullptr && env[0] != '\0') {
        (void) parse_mode(env, config.mode);
    }

    return config;
}

std::vector<llama_moe_hot_cache_entry> llama_moe_hot_cache_weighting::score_observations(
        const std::vector<llama_moe_hot_cache_layer_observation> & observations) {
    return score_observations(observations, default_config());
}

std::vector<llama_moe_hot_cache_entry> llama_moe_hot_cache_weighting::score_observations(
        const std::vector<llama_moe_hot_cache_layer_observation> & observations,
        const llama_moe_hot_cache_weighting_config & config) {
    llama_moe_hot_cache_weighting_config normalized = config;
    normalized.layer_curve = normalize_layer_curve(normalized.layer_curve);
    return weighting_strategy(normalized.mode).score(observations, normalized.layer_curve);
}
