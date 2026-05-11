#include "llama-moe-hot-cache.h"

#include "llama-model.h"
#include "llama-impl.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace {

using json = nlohmann::ordered_json;

static constexpr size_t LLAMA_MOE_HOT_CACHE_MIB = 1024*1024;

static uint64_t key(uint32_t layer, uint32_t expert) {
    return (uint64_t(layer) << 32) | uint64_t(expert);
}

static size_t tensor_expert_bytes(const ggml_tensor * t) {
    if (t == nullptr) {
        return 0;
    }
    if (t->ne[2] <= 0) {
        throw std::runtime_error("MoE expert tensor has invalid expert dimension");
    }
    return ggml_nbytes(t) / size_t(t->ne[2]);
}

static std::vector<llama_moe_hot_cache_expert_size> collect_expert_sizes(const llama_model & model) {
    std::vector<llama_moe_hot_cache_expert_size> result;

    for (uint32_t il = 0; il < model.hparams.n_layer; ++il) {
        const auto & layer = model.layers[il];
        const ggml_tensor * down = layer.ffn_down_exps;
        if (down == nullptr) {
            continue;
        }

        const int64_t n_expert = down->ne[2];
        for (int64_t ex = 0; ex < n_expert; ++ex) {
            size_t bytes = tensor_expert_bytes(down);

            if (layer.ffn_gate_up_exps != nullptr) {
                bytes += tensor_expert_bytes(layer.ffn_gate_up_exps);
            } else {
                bytes += tensor_expert_bytes(layer.ffn_gate_exps);
                bytes += tensor_expert_bytes(layer.ffn_up_exps);
            }

            if (bytes > 0) {
                result.push_back({ il, uint32_t(ex), bytes });
            }
        }
    }

    return result;
}

} // namespace

std::vector<llama_moe_hot_cache_entry> llama_moe_hot_cache_parse_perf_json(const std::string & json_str) {
    json root;
    try {
        root = json::parse(json_str);
    } catch (const std::exception & e) {
        throw std::runtime_error(std::string("failed to parse --moe-hot-cache JSON: ") + e.what());
    }

    if (!root.is_object() || root.value("schema", "") != "llama.cpp.moe_layer_perf.v1") {
        throw std::runtime_error("--moe-hot-cache JSON must use schema llama.cpp.moe_layer_perf.v1");
    }

    if (!root.contains("layers") || !root["layers"].is_array()) {
        throw std::runtime_error("--moe-hot-cache JSON must contain a layers array");
    }

    std::vector<llama_moe_hot_cache_entry> result;
    for (const auto & layer : root["layers"]) {
        if (!layer.is_object() || !layer.contains("layer") || !layer["layer"].is_number_unsigned()) {
            throw std::runtime_error("--moe-hot-cache layer entries must contain an unsigned layer id");
        }
        if (!layer.contains("experts") || !layer["experts"].is_array()) {
            continue;
        }

        const uint32_t il = layer["layer"].get<uint32_t>();
        for (const auto & expert : layer["experts"]) {
            if (!expert.is_array() || expert.size() != 2 ||
                !expert[0].is_number_unsigned() || !expert[1].is_number_unsigned()) {
                throw std::runtime_error("--moe-hot-cache experts must be [expert_id, hit_count] arrays");
            }
            const uint64_t hit_count = expert[1].get<uint64_t>();
            if (hit_count == 0) {
                continue;
            }
            result.push_back({ il, expert[0].get<uint32_t>(), hit_count });
        }
    }

    std::sort(result.begin(), result.end(), [](const auto & a, const auto & b) {
        if (a.hit_count != b.hit_count) {
            return a.hit_count > b.hit_count;
        }
        if (a.layer != b.layer) {
            return a.layer < b.layer;
        }
        return a.expert < b.expert;
    });

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

    for (const auto & entry : observed) {
        const auto it = size_by_expert.find(key(entry.layer, entry.expert));
        if (it == size_by_expert.end()) {
            continue;
        }

        const size_t bytes = it->second;
        if (bytes > budget_bytes || plan.used_bytes + bytes > budget_bytes) {
            continue;
        }

        plan.selected.push_back({ entry.layer, entry.expert, bytes });
        plan.used_bytes += bytes;
    }

    return plan;
}

void llama_moe_hot_cache_init(const llama_model & model, const llama_model_params & params) {
    if (params.moe_hot_cache_max_mib == 0) {
        return;
    }

    if (params.moe_hot_cache_path == nullptr || params.moe_hot_cache_path[0] == '\0') {
        throw std::runtime_error("--moe-hot-cache is required when --moe-hot-cache-max-mib is greater than 0");
    }

    std::ifstream file(params.moe_hot_cache_path);
    if (!file) {
        throw std::runtime_error(std::string("failed to open --moe-hot-cache file: ") + params.moe_hot_cache_path);
    }

    const std::string json_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const auto observed = llama_moe_hot_cache_parse_perf_json(json_str);
    const auto sizes = collect_expert_sizes(model);
    const auto plan = llama_moe_hot_cache_select(observed, sizes, size_t(params.moe_hot_cache_max_mib)*LLAMA_MOE_HOT_CACHE_MIB);

    LLAMA_LOG_INFO("%s: selected %zu/%zu observed experts for hot-cache (%zu/%zu MiB)\n",
            __func__, plan.selected.size(), plan.observed.size(),
            plan.used_bytes/LLAMA_MOE_HOT_CACHE_MIB, plan.budget_bytes/LLAMA_MOE_HOT_CACHE_MIB);

    throw std::runtime_error("--moe-hot-cache is parsed and budgeted, but CUDA hot-cache graph execution is not implemented in this build");
}
