#include "llama-moe-hot-cache-planner.h"

#include "llama-model.h"

#include <algorithm>
#include <limits>
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
            } else {
                bytes += llama_moe_hot_cache_tensor_expert_bytes(layer.ffn_gate_exps);
                bytes += llama_moe_hot_cache_tensor_expert_bytes(layer.ffn_up_exps);
            }

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
    for (const auto & entry : observed) {
        const auto it = size_by_expert.find(key(entry.layer, entry.expert));
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

    throw std::runtime_error("--moe-hot-cache-device-strategy must be one of: warm, hot-even");
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

    for (const auto & entry : observed) {
        const uint64_t entry_key = key(entry.layer, entry.expert);
        if (selected.find(entry_key) != selected.end()) {
            continue;
        }

        const auto size_it = size_by_expert.find(entry_key);
        if (size_it == size_by_expert.end()) {
            continue;
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
            if (select_entry_into_lane(plan.lanes[lane], states[lane], entry, size_it->second)) {
                selected.insert(entry_key);
                break;
            }
        }
    }

    return plan;
}
