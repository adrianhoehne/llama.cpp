#include "llama-moe-hot-cache-updater.h"

#include "llama-moe-hot-cache-builder.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace {

static void add_saturating(uint64_t & dst, uint64_t value) {
    if (dst > std::numeric_limits<uint64_t>::max() - value) {
        dst = std::numeric_limits<uint64_t>::max();
    } else {
        dst += value;
    }
}

} // namespace

std::vector<uint32_t> llama_moe_hot_cache_current_hot_experts(
        const llama_moe_hot_cache_layer & layer) {
    std::vector<uint32_t> result(layer.n_hot, std::numeric_limits<uint32_t>::max());

    for (uint32_t expert = 0; expert < layer.hot_id_map_host.size(); ++expert) {
        const int32_t cache_id = layer.hot_id_map_host[expert];
        if (cache_id >= 0 && uint32_t(cache_id) < result.size()) {
            result[cache_id] = expert;
        }
    }

    result.erase(std::remove(result.begin(), result.end(), std::numeric_limits<uint32_t>::max()), result.end());
    return result;
}

std::vector<llama_moe_hot_cache_replacement_candidate> llama_moe_hot_cache_plan_layer_replacements(
        uint32_t layer,
        const llama_moe_hot_cache_layer & cache_layer,
        const llama_moe_hot_cache_expert_hit_map & expert_hits) {
    std::vector<llama_moe_hot_cache_replacement_candidate> result;

    std::vector<uint32_t> current = llama_moe_hot_cache_current_hot_experts(cache_layer);
    if (current.empty() || expert_hits.empty()) {
        return result;
    }

    std::unordered_set<uint32_t> current_set(current.begin(), current.end());
    std::vector<std::pair<uint32_t, uint64_t>> ranked;
    ranked.reserve(expert_hits.size());
    for (const auto & [expert, hits] : expert_hits) {
        if (expert < cache_layer.n_expert && hits > 0) {
            ranked.push_back({ expert, hits });
        }
    }

    std::sort(ranked.begin(), ranked.end(), [](const auto & a, const auto & b) {
        if (a.second != b.second) {
            return a.second > b.second;
        }
        return a.first < b.first;
    });

    if (ranked.size() > current.size()) {
        ranked.resize(current.size());
    }

    std::unordered_set<uint32_t> desired_set;
    desired_set.reserve(ranked.size());
    for (const auto & [expert, hits] : ranked) {
        GGML_UNUSED(hits);
        desired_set.insert(expert);
    }

    std::vector<std::pair<uint32_t, uint64_t>> to_add;
    for (const auto & [expert, hits] : ranked) {
        if (current_set.find(expert) == current_set.end()) {
            to_add.push_back({ expert, hits });
        }
    }

    std::vector<std::pair<uint32_t, uint64_t>> to_evict;
    for (uint32_t expert : current) {
        if (desired_set.find(expert) == desired_set.end()) {
            const auto it = expert_hits.find(expert);
            to_evict.push_back({ expert, it == expert_hits.end() ? 0 : it->second });
        }
    }

    std::sort(to_evict.begin(), to_evict.end(), [](const auto & a, const auto & b) {
        if (a.second != b.second) {
            return a.second < b.second;
        }
        return a.first < b.first;
    });

    const size_t n_layer_candidates = std::min(to_add.size(), to_evict.size());
    for (size_t i = 0; i < n_layer_candidates; ++i) {
        const uint32_t evict = to_evict[i].first;
        const int32_t cache_id = cache_layer.hot_id_map_host[evict];
        if (cache_id < 0 || uint32_t(cache_id) >= cache_layer.n_hot) {
            continue;
        }

        result.push_back({
                layer,
                evict,
                to_add[i].first,
                uint32_t(cache_id),
                to_add[i].second,
                to_evict[i].second,
            });
    }

    return result;
}

void llama_moe_hot_cache_sort_replacement_candidates(
        std::vector<llama_moe_hot_cache_replacement_candidate> & candidates) {
    std::sort(candidates.begin(), candidates.end(), [](const auto & a, const auto & b) {
        if (a.gain() != b.gain()) {
            return a.gain() > b.gain();
        }
        if (a.layer != b.layer) {
            return a.layer < b.layer;
        }
        return a.add_expert < b.add_expert;
    });
}

size_t llama_moe_hot_cache_update_max_exchange(
        double update_rate,
        size_t hot_experts,
        size_t candidates) {
    const double rate = std::clamp(update_rate, 0.0, 1.0);
    if (rate <= 0.0 || hot_experts == 0) {
        return 0;
    }

    return std::min(candidates, (size_t) std::ceil(rate * (double) hot_experts));
}

llama_moe_hot_cache_update_stats llama_moe_hot_cache_update_from_scored_observations(
        llama_model & model,
        const std::vector<llama_moe_hot_cache_perf_json_layer_slots> & layer_slots,
        const std::vector<llama_moe_hot_cache_entry> & scored_observed,
        double update_rate) {
    llama_moe_hot_cache_update_stats stats;
    stats.update_rate = std::clamp(update_rate, 0.0, 1.0);

    if (!model.moe_hot_cache || !model.moe_hot_cache->active()) {
        return stats;
    }

    const bool has_multi_lane_cache = std::any_of(
            model.moe_hot_cache->layers.begin(),
            model.moe_hot_cache->layers.end(),
            [](const llama_moe_hot_cache_layer & layer) {
                return !layer.lanes.empty();
            });
    if (has_multi_lane_cache) {
        static bool logged = false;
        if (!logged) {
            LLAMA_LOG_WARN("%s: runtime hot-cache replacement is not yet supported for multi-device expert lanes\n", __func__);
            logged = true;
        }
        stats.active = true;
        return stats;
    }

    stats.active = true;

    struct layer_counts {
        llama_moe_hot_cache_expert_hit_map experts;
    };

    std::vector<layer_counts> counts(model.hparams.n_layer());

    for (const auto & entry : scored_observed) {
        if (entry.layer < counts.size() && entry.hit_count > 0) {
            add_saturating(counts[entry.layer].experts[entry.expert], entry.hit_count);
        }
    }

    for (const auto & layer : layer_slots) {
        if (layer.layer >= counts.size()) {
            continue;
        }

        stats.hot_slots += layer.hot_slots;
        stats.cold_slots += layer.cold_slots;
    }

    const uint64_t total_slots = stats.hot_slots + stats.cold_slots;
    stats.hit_rate = total_slots > 0 ? (double) stats.hot_slots / (double) total_slots : 0.0;

    std::vector<llama_moe_hot_cache_replacement_candidate> candidates;

    for (uint32_t il = 0; il < model.moe_hot_cache->layers.size(); ++il) {
        auto & cache_layer = model.moe_hot_cache->layers[il];
        if (!cache_layer.active() || il >= counts.size() || counts[il].experts.empty()) {
            continue;
        }

        const auto current = llama_moe_hot_cache_current_hot_experts(cache_layer);
        if (current.empty()) {
            continue;
        }

        stats.hot_experts += current.size();

        auto layer_candidates = llama_moe_hot_cache_plan_layer_replacements(il, cache_layer, counts[il].experts);
        candidates.insert(candidates.end(), layer_candidates.begin(), layer_candidates.end());
    }

    stats.candidates = candidates.size();
    stats.max_exchange = llama_moe_hot_cache_update_max_exchange(stats.update_rate, stats.hot_experts, candidates.size());

    if (stats.max_exchange == 0) {
        return stats;
    }

    llama_moe_hot_cache_sort_replacement_candidates(candidates);

    std::unordered_set<uint32_t> changed_layers;
    for (size_t i = 0; i < stats.max_exchange; ++i) {
        const auto & candidate = candidates[i];
        const auto & src = model.layers[candidate.layer];
        auto & dst = model.moe_hot_cache->layers[candidate.layer];

        if (candidate.add_expert >= dst.n_expert || candidate.evict_expert >= dst.n_expert) {
            continue;
        }

        const int32_t current_cache_id = dst.hot_id_map_host[candidate.evict_expert];
        if (current_cache_id < 0 || uint32_t(current_cache_id) != candidate.cache_id) {
            continue;
        }
        if (dst.hot_id_map_host[candidate.add_expert] >= 0) {
            continue;
        }

        llama_moe_hot_cache_copy_expert_slice(src.ffn_gate_up_exps, dst.ffn_gate_up_exps, candidate.add_expert, candidate.cache_id);
        llama_moe_hot_cache_copy_expert_slice(src.ffn_gate_exps,    dst.ffn_gate_exps,    candidate.add_expert, candidate.cache_id);
        llama_moe_hot_cache_copy_expert_slice(src.ffn_up_exps,      dst.ffn_up_exps,      candidate.add_expert, candidate.cache_id);
        llama_moe_hot_cache_copy_expert_slice(src.ffn_down_exps,    dst.ffn_down_exps,    candidate.add_expert, candidate.cache_id);
        llama_moe_hot_cache_copy_scale_slice(src.ffn_gate_exps_s,   dst.ffn_gate_exps_s,  candidate.add_expert, candidate.cache_id);
        llama_moe_hot_cache_copy_scale_slice(src.ffn_up_exps_s,     dst.ffn_up_exps_s,    candidate.add_expert, candidate.cache_id);
        llama_moe_hot_cache_copy_scale_slice(src.ffn_down_exps_s,   dst.ffn_down_exps_s,  candidate.add_expert, candidate.cache_id);

        dst.hot_id_map_host[candidate.evict_expert] = -1;
        dst.hot_id_map_host[candidate.add_expert] = int32_t(candidate.cache_id);

        llama_moe_hot_cache_set_tensor_i32_1d(dst.hot_id_map, candidate.evict_expert, int32_t(dst.n_hot));
        llama_moe_hot_cache_set_tensor_i32_1d(dst.hot_id_map, candidate.add_expert, int32_t(candidate.cache_id));
        llama_moe_hot_cache_set_tensor_f32_1d(dst.hot_mask, candidate.evict_expert, 0.0f);
        llama_moe_hot_cache_set_tensor_f32_1d(dst.hot_mask, candidate.add_expert, 1.0f);
        llama_moe_hot_cache_set_tensor_f32_1d(dst.cold_mask, candidate.evict_expert, 1.0f);
        llama_moe_hot_cache_set_tensor_f32_1d(dst.cold_mask, candidate.add_expert, 0.0f);

        stats.exchanged++;
        changed_layers.insert(candidate.layer);
    }

    stats.layers_changed = changed_layers.size();
    return stats;
}
