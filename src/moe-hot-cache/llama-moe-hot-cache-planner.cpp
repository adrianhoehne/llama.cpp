#include "llama-moe-hot-cache-planner.h"

#include "llama-model.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace {

static uint64_t key(uint32_t layer, uint32_t expert) {
    return (uint64_t(layer) << 32) | uint64_t(expert);
}

struct lane_state {
    std::unordered_set<uint32_t> active_layers;
    std::unordered_map<uint32_t, size_t> selected_by_layer;
};

static std::vector<llama_moe_hot_cache_entry> random_fill_entries(
        const std::vector<llama_moe_hot_cache_expert_size> & sizes,
        const std::unordered_set<uint64_t> & selected);

static bool env_flag_enabled_by_default(const char * name, bool default_value) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    const std::string v(value);
    return v != "0" && v != "false" && v != "False" && v != "off" && v != "OFF";
}

static bool select_entry_into_lane(
        llama_moe_hot_cache_plan & lane,
        lane_state & state,
        const llama_moe_hot_cache_entry & entry,
        size_t bytes) {
    size_t cost = bytes;
    if (state.active_layers.find(entry.layer) == state.active_layers.end()) {
        if (cost > std::numeric_limits<size_t>::max() - bytes) {
            return false;
        }
        cost += bytes;
    }

    if (cost > lane.budget_bytes || lane.used_bytes > lane.budget_bytes - cost) {
        return false;
    }

    lane.selected.push_back({ entry.layer, entry.expert, bytes });
    lane.used_bytes += cost;
    state.active_layers.insert(entry.layer);
    state.selected_by_layer[entry.layer]++;
    return true;
}

static size_t best_hot_even_lane(
        const std::vector<llama_moe_hot_cache_plan> & lanes,
        const std::vector<lane_state> & states,
        const llama_moe_hot_cache_entry & entry,
        const std::vector<size_t> & candidates) {
    size_t best = std::numeric_limits<size_t>::max();
    for (size_t lane : candidates) {
        if (lane >= lanes.size()) {
            continue;
        }

        if (best == std::numeric_limits<size_t>::max()) {
            best = lane;
            continue;
        }

        const size_t lane_layer_count = states[lane].selected_by_layer.count(entry.layer)
            ? states[lane].selected_by_layer.at(entry.layer)
            : 0;
        const size_t best_layer_count = states[best].selected_by_layer.count(entry.layer)
            ? states[best].selected_by_layer.at(entry.layer)
            : 0;

        if (lane_layer_count != best_layer_count) {
            if (lane_layer_count < best_layer_count) {
                best = lane;
            }
            continue;
        }

        if (lanes[lane].used_bytes != lanes[best].used_bytes) {
            if (lanes[lane].used_bytes < lanes[best].used_bytes) {
                best = lane;
            }
            continue;
        }

        if (lane < best) {
            best = lane;
        }
    }

    return best;
}

// Return all MoE layers in model order so even-split can create contiguous layer bands.
static std::vector<uint32_t> sorted_layers_from_sizes(
        const std::vector<llama_moe_hot_cache_expert_size> & sizes) {
    std::vector<uint32_t> layers;
    layers.reserve(sizes.size());
    for (const auto & size : sizes) {
        layers.push_back(size.layer);
    }

    std::sort(layers.begin(), layers.end());
    layers.erase(std::unique(layers.begin(), layers.end()), layers.end());
    return layers;
}

// Split layers into contiguous lane-owned bands proportional to each lane budget.
static std::vector<std::vector<uint32_t>> even_split_layers_by_lane(
        const std::vector<llama_moe_hot_cache_expert_size> & sizes,
        const std::vector<size_t> & lane_budget_bytes) {
    std::vector<std::vector<uint32_t>> result(lane_budget_bytes.size());
    const std::vector<uint32_t> layers = sorted_layers_from_sizes(sizes);
    if (layers.empty()) {
        return result;
    }

    size_t total_budget = 0;
    for (const size_t budget : lane_budget_bytes) {
        if (total_budget > std::numeric_limits<size_t>::max() - budget) {
            total_budget = std::numeric_limits<size_t>::max();
            break;
        }
        total_budget += budget;
    }
    if (total_budget == 0) {
        return result;
    }

    size_t begin = 0;
    long double cumulative_budget = 0.0L;
    for (size_t lane = 0; lane < lane_budget_bytes.size(); ++lane) {
        cumulative_budget += (long double) lane_budget_bytes[lane];
        size_t end = layers.size();
        if (lane + 1 < lane_budget_bytes.size()) {
            const long double target =
                cumulative_budget * (long double) layers.size() / (long double) total_budget;
            end = (size_t) (target + 0.5L);
        }

        end = std::max(end, begin);
        end = std::min(end, layers.size());
        result[lane].assign(
                layers.begin() + (std::ptrdiff_t) begin,
                layers.begin() + (std::ptrdiff_t) end);
        begin = end;
    }

    return result;
}

// Select lane-local experts round-robin by layer so each owned layer gets similar cache depth.
static void select_even_split_lane_entries(
        llama_moe_hot_cache_multi_plan & plan,
        std::vector<lane_state> & states,
        std::unordered_set<uint64_t> & selected,
        const std::unordered_map<uint64_t, size_t> & size_by_expert,
        size_t lane,
        const std::vector<uint32_t> & layers,
        std::unordered_map<uint32_t, std::vector<llama_moe_hot_cache_entry>> & entries_by_layer) {
    std::unordered_map<uint32_t, size_t> offsets;
    bool made_progress = true;
    while (made_progress) {
        made_progress = false;
        for (const uint32_t layer : layers) {
            auto entries_it = entries_by_layer.find(layer);
            if (entries_it == entries_by_layer.end()) {
                continue;
            }

            auto & entries = entries_it->second;
            size_t & offset = offsets[layer];
            while (offset < entries.size()) {
                const llama_moe_hot_cache_entry & entry = entries[offset++];
                const uint64_t entry_key = key(entry.layer, entry.expert);
                if (selected.find(entry_key) != selected.end()) {
                    continue;
                }

                const auto size_it = size_by_expert.find(entry_key);
                if (size_it == size_by_expert.end()) {
                    continue;
                }

                if (select_entry_into_lane(plan.lanes[lane], states[lane], entry, size_it->second)) {
                    selected.insert(entry_key);
                    made_progress = true;
                    break;
                }
            }
        }
    }
}

// Build an even-split plan: one lane owns one contiguous layer band, then fills that band evenly.
static void select_even_split_multi_plan(
        llama_moe_hot_cache_multi_plan & plan,
        std::vector<lane_state> & states,
        std::unordered_set<uint64_t> & selected,
        const std::vector<llama_moe_hot_cache_entry> & observed,
        const std::vector<llama_moe_hot_cache_expert_size> & sizes,
        const std::vector<size_t> & lane_budget_bytes,
        const std::unordered_map<uint64_t, size_t> & size_by_expert) {
    const std::vector<std::vector<uint32_t>> layers_by_lane =
        even_split_layers_by_lane(sizes, lane_budget_bytes);

    std::unordered_map<uint32_t, size_t> lane_by_layer;
    for (size_t lane = 0; lane < layers_by_lane.size(); ++lane) {
        for (const uint32_t layer : layers_by_lane[lane]) {
            lane_by_layer[layer] = lane;
        }
    }

    std::vector<std::unordered_map<uint32_t, std::vector<llama_moe_hot_cache_entry>>> entries_by_lane_layer(
            plan.lanes.size());
    std::unordered_set<uint64_t> queued;
    for (const auto & entry : observed) {
        const uint64_t entry_key = key(entry.layer, entry.expert);
        if (queued.find(entry_key) != queued.end() || size_by_expert.find(entry_key) == size_by_expert.end()) {
            continue;
        }

        const auto lane_it = lane_by_layer.find(entry.layer);
        if (lane_it == lane_by_layer.end()) {
            continue;
        }

        entries_by_lane_layer[lane_it->second][entry.layer].push_back(entry);
        queued.insert(entry_key);
    }

    if (env_flag_enabled_by_default("LLAMA_MOE_HOT_CACHE_FILL_RANDOM", true)) {
        for (const auto & entry : random_fill_entries(sizes, queued)) {
            const uint64_t entry_key = key(entry.layer, entry.expert);
            const auto lane_it = lane_by_layer.find(entry.layer);
            if (lane_it == lane_by_layer.end() || size_by_expert.find(entry_key) == size_by_expert.end()) {
                continue;
            }

            entries_by_lane_layer[lane_it->second][entry.layer].push_back(entry);
            queued.insert(entry_key);
        }
    }

    for (size_t lane = 0; lane < plan.lanes.size(); ++lane) {
        select_even_split_lane_entries(
                plan,
                states,
                selected,
                size_by_expert,
                lane,
                layers_by_lane[lane],
                entries_by_lane_layer[lane]);
    }
}

// Select one expert into the best available lane for the configured placement strategy.
static bool select_entry_into_multi_plan(
        llama_moe_hot_cache_multi_plan & plan,
        std::vector<lane_state> & states,
        std::unordered_set<uint64_t> & selected,
        const llama_moe_hot_cache_entry & entry,
        size_t bytes,
        llama_moe_hot_cache_device_strategy strategy) {
    const uint64_t entry_key = key(entry.layer, entry.expert);
    if (selected.find(entry_key) != selected.end()) {
        return false;
    }

    std::vector<size_t> lane_order(plan.lanes.size());
    for (size_t i = 0; i < lane_order.size(); ++i) {
        lane_order[i] = i;
    }

    if (strategy == llama_moe_hot_cache_device_strategy::hot_even) {
        std::sort(lane_order.begin(), lane_order.end(), [&](size_t a, size_t b) {
            const size_t a_count = states[a].selected_by_layer.count(entry.layer)
                ? states[a].selected_by_layer.at(entry.layer)
                : 0;
            const size_t b_count = states[b].selected_by_layer.count(entry.layer)
                ? states[b].selected_by_layer.at(entry.layer)
                : 0;
            if (a_count != b_count) {
                return a_count < b_count;
            }
            if (plan.lanes[a].used_bytes != plan.lanes[b].used_bytes) {
                return plan.lanes[a].used_bytes < plan.lanes[b].used_bytes;
            }
            return a < b;
        });

        const size_t preferred = best_hot_even_lane(plan.lanes, states, entry, lane_order);
        if (preferred != std::numeric_limits<size_t>::max()) {
            lane_order.erase(std::remove(lane_order.begin(), lane_order.end(), preferred), lane_order.end());
            lane_order.insert(lane_order.begin(), preferred);
        }
    }

    for (size_t lane : lane_order) {
        if (select_entry_into_lane(plan.lanes[lane], states[lane], entry, bytes)) {
            selected.insert(entry_key);
            return true;
        }
    }

    return false;
}

// Append deterministic random fallback candidates so unused budget can hold coverage experts.
static std::vector<llama_moe_hot_cache_entry> random_fill_entries(
        const std::vector<llama_moe_hot_cache_expert_size> & sizes,
        const std::unordered_set<uint64_t> & selected) {
    std::vector<llama_moe_hot_cache_entry> result;
    result.reserve(sizes.size());

    for (const auto & size : sizes) {
        if (selected.find(key(size.layer, size.expert)) == selected.end()) {
            result.push_back({ size.layer, size.expert, 0 });
        }
    }

    std::mt19937 rng(0x6d6f65u);
    std::shuffle(result.begin(), result.end(), rng);
    return result;
}

} // namespace

size_t llama_moe_hot_cache_multi_plan::selected_count() const {
    size_t result = 0;
    for (const auto & lane : lanes) {
        result += lane.selected.size();
    }
    return result;
}

size_t llama_moe_hot_cache_multi_plan::used_bytes() const {
    size_t result = 0;
    for (const auto & lane : lanes) {
        if (result > std::numeric_limits<size_t>::max() - lane.used_bytes) {
            return std::numeric_limits<size_t>::max();
        }
        result += lane.used_bytes;
    }
    return result;
}

size_t llama_moe_hot_cache_multi_plan::budget_bytes() const {
    size_t result = 0;
    for (const auto & lane : lanes) {
        if (result > std::numeric_limits<size_t>::max() - lane.budget_bytes) {
            return std::numeric_limits<size_t>::max();
        }
        result += lane.budget_bytes;
    }
    return result;
}

size_t llama_moe_hot_cache_tensor_expert_bytes(const ggml_tensor * t) {
    if (t == nullptr) {
        return 0;
    }
    if (t->ne[2] <= 0) {
        throw std::runtime_error("MoE expert tensor has invalid expert dimension");
    }
    return ggml_nbytes(t) / size_t(t->ne[2]);
}

size_t llama_moe_hot_cache_tensor_expert_bias_bytes(const ggml_tensor * t) {
    if (t == nullptr) {
        return 0;
    }
    if (t->ne[1] <= 0) {
        throw std::runtime_error("MoE expert bias tensor has invalid expert dimension");
    }
    return ggml_nbytes(t) / size_t(t->ne[1]);
}

std::vector<llama_moe_hot_cache_expert_size> llama_moe_hot_cache_collect_expert_sizes(
        const llama_model & model) {
    std::vector<llama_moe_hot_cache_expert_size> result;

    for (uint32_t il = 0; il < model.hparams.n_layer(); ++il) {
        const auto & layer = model.layers[il];
        const ggml_tensor * down = layer.ffn_down_exps;
        if (down == nullptr) {
            continue;
        }

        const int64_t n_expert = down->ne[2];
        for (int64_t ex = 0; ex < n_expert; ++ex) {
            size_t bytes = llama_moe_hot_cache_tensor_expert_bytes(down);

            if (layer.ffn_gate_up_exps != nullptr) {
                bytes += llama_moe_hot_cache_tensor_expert_bytes(layer.ffn_gate_up_exps);
                bytes += llama_moe_hot_cache_tensor_expert_bias_bytes(layer.ffn_gate_up_exps_b);
            } else {
                bytes += llama_moe_hot_cache_tensor_expert_bytes(layer.ffn_gate_exps);
                bytes += llama_moe_hot_cache_tensor_expert_bytes(layer.ffn_up_exps);
                bytes += llama_moe_hot_cache_tensor_expert_bias_bytes(layer.ffn_gate_exps_b);
                bytes += llama_moe_hot_cache_tensor_expert_bias_bytes(layer.ffn_up_exps_b);
            }
            bytes += llama_moe_hot_cache_tensor_expert_bias_bytes(layer.ffn_down_exps_b);

            if (bytes > 0) {
                result.push_back({ il, uint32_t(ex), bytes });
            }
        }
    }

    return result;
}

llama_moe_hot_cache_plan llama_moe_hot_cache_select(
        const std::vector<llama_moe_hot_cache_entry> & observed,
        const std::vector<llama_moe_hot_cache_expert_size> & sizes,
        size_t budget_bytes) {
    llama_moe_hot_cache_plan plan;
    plan.observed = observed;
    plan.budget_bytes = budget_bytes;

    std::unordered_map<uint64_t, size_t> size_by_expert;
    size_by_expert.reserve(sizes.size());
    for (const auto & size : sizes) {
        size_by_expert[key(size.layer, size.expert)] = size.bytes;
    }

    std::unordered_set<uint32_t> active_layers;
    std::unordered_set<uint64_t> selected;
    for (const auto & entry : observed) {
        const uint64_t entry_key = key(entry.layer, entry.expert);
        const auto it = size_by_expert.find(entry_key);
        if (it == size_by_expert.end()) {
            continue;
        }

        const size_t bytes = it->second;
        size_t cost = bytes;
        if (active_layers.find(entry.layer) == active_layers.end()) {
            if (cost > std::numeric_limits<size_t>::max() - bytes) {
                continue;
            }
            cost += bytes; // final entry in each active layer is a zero dummy expert
        }

        if (cost > budget_bytes || plan.used_bytes > budget_bytes - cost) {
            continue;
        }

        plan.selected.push_back({ entry.layer, entry.expert, bytes });
        plan.used_bytes += cost;
        active_layers.insert(entry.layer);
        selected.insert(entry_key);
    }

    if (env_flag_enabled_by_default("LLAMA_MOE_HOT_CACHE_FILL_RANDOM", true)) {
        for (const auto & entry : random_fill_entries(sizes, selected)) {
            const uint64_t entry_key = key(entry.layer, entry.expert);
            const auto it = size_by_expert.find(entry_key);
            if (it == size_by_expert.end()) {
                continue;
            }

            const size_t bytes = it->second;
            size_t cost = bytes;
            if (active_layers.find(entry.layer) == active_layers.end()) {
                if (cost > std::numeric_limits<size_t>::max() - bytes) {
                    continue;
                }
                cost += bytes;
            }

            if (cost > budget_bytes || plan.used_bytes > budget_bytes - cost) {
                continue;
            }

            plan.selected.push_back({ entry.layer, entry.expert, bytes });
            plan.used_bytes += cost;
            active_layers.insert(entry.layer);
            selected.insert(entry_key);
        }
    }

    return plan;
}

llama_moe_hot_cache_device_strategy llama_moe_hot_cache_parse_device_strategy(
        const char * name) {
    if (name == nullptr || name[0] == '\0' || std::string(name) == "warm") {
        return llama_moe_hot_cache_device_strategy::warm;
    }
    if (std::string(name) == "hot-even") {
        return llama_moe_hot_cache_device_strategy::hot_even;
    }
    if (std::string(name) == "even-split") {
        return llama_moe_hot_cache_device_strategy::even_split;
    }

    throw std::runtime_error("--moe-hot-cache-device-strategy must be one of: warm, hot-even, even-split");
}

llama_moe_hot_cache_multi_plan llama_moe_hot_cache_select_multi(
        const std::vector<llama_moe_hot_cache_entry> & observed,
        const std::vector<llama_moe_hot_cache_expert_size> & sizes,
        const std::vector<size_t> & lane_budget_bytes,
        llama_moe_hot_cache_device_strategy strategy) {
    if (lane_budget_bytes.empty() || lane_budget_bytes.size() > LLAMA_MOE_HOT_CACHE_MAX_EXPERT_LANES) {
        throw std::runtime_error("MoE hot-cache multi-device planner requires 1..3 expert lanes");
    }

    llama_moe_hot_cache_multi_plan plan;
    plan.observed = observed;
    plan.lanes.resize(lane_budget_bytes.size());
    for (size_t i = 0; i < plan.lanes.size(); ++i) {
        plan.lanes[i].observed = observed;
        plan.lanes[i].budget_bytes = lane_budget_bytes[i];
    }

    std::unordered_map<uint64_t, size_t> size_by_expert;
    size_by_expert.reserve(sizes.size());
    for (const auto & size : sizes) {
        size_by_expert[key(size.layer, size.expert)] = size.bytes;
    }

    std::unordered_set<uint64_t> selected;
    std::vector<lane_state> states(plan.lanes.size());

    if (strategy == llama_moe_hot_cache_device_strategy::even_split) {
        select_even_split_multi_plan(
                plan,
                states,
                selected,
                observed,
                sizes,
                lane_budget_bytes,
                size_by_expert);
        return plan;
    }

    for (const auto & entry : observed) {
        const uint64_t entry_key = key(entry.layer, entry.expert);
        if (selected.find(entry_key) != selected.end()) {
            continue;
        }

        const auto size_it = size_by_expert.find(entry_key);
        if (size_it == size_by_expert.end()) {
            continue;
        }

        select_entry_into_multi_plan(plan, states, selected, entry, size_it->second, strategy);
    }

    if (env_flag_enabled_by_default("LLAMA_MOE_HOT_CACHE_FILL_RANDOM", true)) {
        for (const auto & entry : random_fill_entries(sizes, selected)) {
            const auto size_it = size_by_expert.find(key(entry.layer, entry.expert));
            if (size_it == size_by_expert.end()) {
                continue;
            }

            select_entry_into_multi_plan(plan, states, selected, entry, size_it->second, strategy);
        }
    }

    return plan;
}
