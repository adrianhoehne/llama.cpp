#include "llama-moe-hot-cache-planner.h"

#include "llama-model.h"

#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {

static uint64_t key(uint32_t layer, uint32_t expert) {
    return (uint64_t(layer) << 32) | uint64_t(expert);
}

} // namespace

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
