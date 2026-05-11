#include "llama-moe-hot-cache.h"

#include "llama-model.h"
#include "llama-impl.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
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

static ggml_backend_buffer_type_t select_gpu_buft() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == nullptr) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    }
    if (dev == nullptr) {
        throw std::runtime_error("--moe-hot-cache requires a GPU backend device");
    }

    return ggml_backend_dev_buffer_type(dev);
}

static ggml_tensor * new_tensor_like_experts(
        ggml_context * ctx,
        const ggml_tensor * src,
        int64_t n_cache,
        const char * name) {
    if (src == nullptr) {
        return nullptr;
    }

    ggml_tensor * dst = ggml_new_tensor_3d(ctx, src->type, src->ne[0], src->ne[1], n_cache);
    ggml_set_name(dst, name);
    return dst;
}

static void zero_tensor(ggml_tensor * t) {
    std::vector<uint8_t> zeros(ggml_nbytes(t), 0);
    ggml_backend_tensor_set(t, zeros.data(), 0, zeros.size());
}

static void copy_expert_slice(const ggml_tensor * src, ggml_tensor * dst, uint32_t src_expert, uint32_t dst_expert) {
    if (src == nullptr || dst == nullptr) {
        return;
    }

    const size_t bytes = tensor_expert_bytes(src);
    std::vector<uint8_t> buf(bytes);
    ggml_backend_tensor_get(src, buf.data(), src->nb[2]*src_expert, bytes);
    ggml_backend_tensor_set(dst, buf.data(), dst->nb[2]*dst_expert, bytes);
}

static ggml_tensor * new_tensor_like_scale(
        ggml_context * ctx,
        const ggml_tensor * src,
        int64_t n_cache,
        const char * name) {
    if (src == nullptr) {
        return nullptr;
    }

    ggml_tensor * dst = ggml_new_tensor_1d(ctx, src->type, n_cache);
    ggml_set_name(dst, name);
    return dst;
}

static void copy_scale_slice(const ggml_tensor * src, ggml_tensor * dst, uint32_t src_expert, uint32_t dst_expert) {
    if (src == nullptr || dst == nullptr) {
        return;
    }

    const size_t bytes = ggml_nbytes(src) / size_t(src->ne[0]);
    std::vector<uint8_t> buf(bytes);
    ggml_backend_tensor_get(src, buf.data(), src->nb[0]*src_expert, bytes);
    ggml_backend_tensor_set(dst, buf.data(), dst->nb[0]*dst_expert, bytes);
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

void llama_moe_hot_cache_init(llama_model & model, const llama_model_params & params) {
    if (params.moe_hot_cache_max_mib == 0) {
        return;
    }

    if (params.no_alloc) {
        LLAMA_LOG_INFO("%s: skipping hot-cache build during no-alloc model load\n", __func__);
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

    if (plan.selected.empty()) {
        LLAMA_LOG_WARN("%s: no experts selected; disabling hot-cache\n", __func__);
        return;
    }

    auto cache = std::make_unique<llama_moe_hot_cache>();
    cache->layers.resize(model.hparams.n_layer);

    std::vector<std::vector<uint32_t>> selected_by_layer(model.hparams.n_layer);
    for (const auto & selected : plan.selected) {
        if (selected.layer < selected_by_layer.size()) {
            selected_by_layer[selected.layer].push_back(selected.expert);
        }
    }

    size_t n_tensors = 0;
    for (uint32_t il = 0; il < selected_by_layer.size(); ++il) {
        if (selected_by_layer[il].empty()) {
            continue;
        }

        const auto & layer = model.layers[il];
        const int64_t n_cache = int64_t(selected_by_layer[il].size()) + 1; // final entry is a zero dummy expert

        n_tensors += 4; // map + hot mask + cold mask + down
        n_tensors += layer.ffn_gate_up_exps != nullptr ? 1 : 2;
        n_tensors += layer.ffn_gate_exps_s != nullptr ? 1 : 0;
        n_tensors += layer.ffn_up_exps_s   != nullptr ? 1 : 0;
        n_tensors += layer.ffn_down_exps_s != nullptr ? 1 : 0;

        cache->layers[il].n_hot = selected_by_layer[il].size();
        cache->layers[il].n_expert = layer.ffn_down_exps ? layer.ffn_down_exps->ne[2] : 0;

        GGML_UNUSED(n_cache);
    }

    // Reserve extra space per tensor for internal ggml bookkeeping
    // (hash map entries, linked-list pointers, etc. beyond ggml_tensor_overhead())
    static constexpr size_t EXTRA_PER_TENSOR = 64;
    const size_t n_extra_tensors = std::max<size_t>(64, n_tensors / 4);
    const size_t ctx_mem_size = ggml_tensor_overhead() * (n_tensors + n_extra_tensors)
                              + EXTRA_PER_TENSOR * n_tensors;

    ggml_init_params ctx_params = {
        /*.mem_size   =*/ ctx_mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx { ggml_init(ctx_params) };
    if (!ctx) {
        throw std::runtime_error("failed to create MoE hot-cache ggml context");
    }

    for (uint32_t il = 0; il < selected_by_layer.size(); ++il) {
        const auto & experts = selected_by_layer[il];
        if (experts.empty()) {
            continue;
        }

        const auto & src = model.layers[il];
        auto & dst = cache->layers[il];
        const int64_t n_cache = int64_t(experts.size()) + 1;
        const int64_t n_expert = src.ffn_down_exps->ne[2];

        dst.ffn_gate_up_exps = new_tensor_like_experts(ctx.get(), src.ffn_gate_up_exps, n_cache, format("blk.%u.ffn_gate_up_exps.hot_cache", il).c_str());
        dst.ffn_gate_exps    = new_tensor_like_experts(ctx.get(), src.ffn_gate_exps,    n_cache, format("blk.%u.ffn_gate_exps.hot_cache",    il).c_str());
        dst.ffn_up_exps      = new_tensor_like_experts(ctx.get(), src.ffn_up_exps,      n_cache, format("blk.%u.ffn_up_exps.hot_cache",      il).c_str());
        dst.ffn_down_exps    = new_tensor_like_experts(ctx.get(), src.ffn_down_exps,    n_cache, format("blk.%u.ffn_down_exps.hot_cache",    il).c_str());
        dst.ffn_gate_exps_s  = new_tensor_like_scale  (ctx.get(), src.ffn_gate_exps_s,  n_cache, format("blk.%u.ffn_gate_exps_s.hot_cache",  il).c_str());
        dst.ffn_up_exps_s    = new_tensor_like_scale  (ctx.get(), src.ffn_up_exps_s,    n_cache, format("blk.%u.ffn_up_exps_s.hot_cache",    il).c_str());
        dst.ffn_down_exps_s  = new_tensor_like_scale  (ctx.get(), src.ffn_down_exps_s,  n_cache, format("blk.%u.ffn_down_exps_s.hot_cache",  il).c_str());

        dst.hot_id_map = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, n_expert);
        ggml_format_name(dst.hot_id_map, "blk.%u.moe_hot_id_map", il);
        dst.hot_mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, n_expert);
        ggml_format_name(dst.hot_mask, "blk.%u.moe_hot_mask", il);
        dst.cold_mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, n_expert);
        ggml_format_name(dst.cold_mask, "blk.%u.moe_cold_mask", il);
    }

    ggml_backend_buffer_type_t buft = select_gpu_buft();
    ggml_backend_buffer_ptr buf { ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft) };
    if (!buf) {
        throw std::runtime_error("failed to allocate MoE hot-cache buffer");
    }
    ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    for (uint32_t il = 0; il < selected_by_layer.size(); ++il) {
        const auto & experts = selected_by_layer[il];
        if (experts.empty()) {
            continue;
        }

        const auto & src = model.layers[il];
        auto & dst = cache->layers[il];
        const uint32_t dummy_id = experts.size();
        const uint32_t n_expert = dst.n_expert;

        if (dst.ffn_gate_up_exps) { zero_tensor(dst.ffn_gate_up_exps); }
        if (dst.ffn_gate_exps)    { zero_tensor(dst.ffn_gate_exps); }
        if (dst.ffn_up_exps)      { zero_tensor(dst.ffn_up_exps); }
        if (dst.ffn_down_exps)    { zero_tensor(dst.ffn_down_exps); }
        if (dst.ffn_gate_exps_s)  { zero_tensor(dst.ffn_gate_exps_s); }
        if (dst.ffn_up_exps_s)    { zero_tensor(dst.ffn_up_exps_s); }
        if (dst.ffn_down_exps_s)  { zero_tensor(dst.ffn_down_exps_s); }

        std::vector<int32_t> hot_id_map(n_expert, int32_t(dummy_id));
        std::vector<float> hot_mask(n_expert, 0.0f);
        std::vector<float> cold_mask(n_expert, 1.0f);

        for (uint32_t cache_id = 0; cache_id < experts.size(); ++cache_id) {
            const uint32_t expert = experts[cache_id];
            if (expert >= n_expert) {
                continue;
            }

            hot_id_map[expert] = int32_t(cache_id);
            hot_mask[expert] = 1.0f;
            cold_mask[expert] = 0.0f;

            copy_expert_slice(src.ffn_gate_up_exps, dst.ffn_gate_up_exps, expert, cache_id);
            copy_expert_slice(src.ffn_gate_exps,    dst.ffn_gate_exps,    expert, cache_id);
            copy_expert_slice(src.ffn_up_exps,      dst.ffn_up_exps,      expert, cache_id);
            copy_expert_slice(src.ffn_down_exps,    dst.ffn_down_exps,    expert, cache_id);
            copy_scale_slice(src.ffn_gate_exps_s,   dst.ffn_gate_exps_s,  expert, cache_id);
            copy_scale_slice(src.ffn_up_exps_s,     dst.ffn_up_exps_s,    expert, cache_id);
            copy_scale_slice(src.ffn_down_exps_s,   dst.ffn_down_exps_s,  expert, cache_id);
        }

        ggml_backend_tensor_set(dst.hot_id_map, hot_id_map.data(), 0, hot_id_map.size()*sizeof(hot_id_map[0]));
        ggml_backend_tensor_set(dst.hot_mask,   hot_mask.data(),   0, hot_mask.size()*sizeof(hot_mask[0]));
        ggml_backend_tensor_set(dst.cold_mask,  cold_mask.data(),  0, cold_mask.size()*sizeof(cold_mask[0]));
    }

    LLAMA_LOG_INFO("%s: %12s hot-cache buffer size = %8.2f MiB\n",
            __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get())/1024.0/1024.0);

    cache->bufs.emplace_back(std::move(buf));
    cache->ctxs.emplace_back(std::move(ctx));
    model.moe_hot_cache = std::move(cache);
}
