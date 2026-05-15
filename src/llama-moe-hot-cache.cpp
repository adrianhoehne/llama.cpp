#include "llama-moe-hot-cache.h"

#include "llama-model.h"
#include "llama-impl.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace {

using json = nlohmann::ordered_json;

static constexpr size_t LLAMA_MOE_HOT_CACHE_MIB = 1024*1024;
static constexpr double LLAMA_MOE_HOT_CACHE_LAYER_WEIGHT_DAMPING = 0.75;
static constexpr double LLAMA_MOE_HOT_CACHE_LAYER_WEIGHT_MIN = 0.80;
static constexpr double LLAMA_MOE_HOT_CACHE_LAYER_WEIGHT_MAX = 1.25;
static constexpr double LLAMA_MOE_HOT_CACHE_HOT_STICKY_BONUS = 0.05;

static bool llama_moe_hot_cache_hot_dummy_padding() {
    static const bool enabled = []() {
        const char * env = std::getenv("LLAMA_MOE_HOT_CACHE_HOT_DUMMY_PADDING");
        return env == nullptr || env[0] == '\0' ||
               (std::strcmp(env, "0") != 0 && std::strcmp(env, "off") != 0 && std::strcmp(env, "false") != 0);
    }();

    return enabled;
}

struct expert_counts {
    uint64_t hot = 0;
    uint64_t cold = 0;
    uint64_t raw = 0;
};

struct layer_observation {
    uint32_t layer = 0;
    std::unordered_map<uint32_t, expert_counts> experts;
    bool has_branch_counts = false;
    double wait_per_cold_slot = 0.0;
};

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

static void add_saturating(uint64_t & dst, uint64_t value) {
    if (dst > std::numeric_limits<uint64_t>::max() - value) {
        dst = std::numeric_limits<uint64_t>::max();
    } else {
        dst += value;
    }
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

    std::vector<layer_observation> observations;
    for (const auto & layer : root["layers"]) {
        if (!layer.is_object() || !layer.contains("layer") || !layer["layer"].is_number_unsigned()) {
            throw std::runtime_error("--moe-hot-cache layer entries must contain an unsigned layer id");
        }

        layer_observation obs;
        obs.layer = layer["layer"].get<uint32_t>();

        const auto add_expert_counts = [&](const char * field_name, uint64_t expert_counts::* member) {
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
                add_saturating(obs.experts[expert_id].*member, hit_count);
                added_any = true;
            }

            return added_any;
        };

        obs.has_branch_counts =
            add_expert_counts("hot_experts",  &expert_counts::hot) |
            add_expert_counts("cold_experts", &expert_counts::cold);

        if (!obs.has_branch_counts && !add_expert_counts("experts", &expert_counts::raw)) {
            continue;
        }

        double cold_slots_per_call = layer.value("cold_slots_per_call", 0.0);
        if (cold_slots_per_call <= 0.0) {
            const uint64_t calls = layer.value("calls", uint64_t(0));
            const uint64_t cold_slots_total = layer.value("cold_slots_total", uint64_t(0));
            if (calls > 0 && cold_slots_total > 0) {
                cold_slots_per_call = (double) cold_slots_total / (double) calls;
            }
        }

        if (cold_slots_per_call > 0.0) {
            double wait_per_call = layer.value("parallel_join_wait_time_per_call_us", 0.0);
            if (wait_per_call <= 0.0) {
                const double cold_lane_per_call = layer.value("parallel_cold_lane_wall_time_per_call_us", 0.0);
                const double hot_lane_per_call  = layer.value("parallel_hot_lane_wall_time_per_call_us", 0.0);
                wait_per_call = std::max(0.0, cold_lane_per_call - hot_lane_per_call);
            }

            if (wait_per_call > 0.0) {
                obs.wait_per_cold_slot = wait_per_call / cold_slots_per_call;
            }
        }

        observations.push_back(std::move(obs));
    }

    double wait_per_cold_slot_sum = 0.0;
    int wait_per_cold_slot_count = 0;
    for (const auto & obs : observations) {
        if (obs.wait_per_cold_slot > 0.0) {
            wait_per_cold_slot_sum += obs.wait_per_cold_slot;
            ++wait_per_cold_slot_count;
        }
    }

    const double avg_wait_per_cold_slot = wait_per_cold_slot_count > 0
        ? wait_per_cold_slot_sum / double(wait_per_cold_slot_count)
        : 0.0;

    std::vector<llama_moe_hot_cache_entry> result;
    for (const auto & obs : observations) {
        double layer_weight = 1.0;
        if (avg_wait_per_cold_slot > 0.0 && obs.wait_per_cold_slot > 0.0) {
            const double relative_wait = obs.wait_per_cold_slot / avg_wait_per_cold_slot;
            layer_weight = 1.0 + LLAMA_MOE_HOT_CACHE_LAYER_WEIGHT_DAMPING*(relative_wait - 1.0);
            layer_weight = std::clamp(layer_weight, LLAMA_MOE_HOT_CACHE_LAYER_WEIGHT_MIN, LLAMA_MOE_HOT_CACHE_LAYER_WEIGHT_MAX);
        }

        for (const auto & [expert, counts] : obs.experts) {
            uint64_t total_hits = counts.raw;
            if (obs.has_branch_counts) {
                total_hits = counts.hot;
                add_saturating(total_hits, counts.cold);
            }
            if (total_hits == 0) {
                continue;
            }

            long double weighted = (long double) total_hits * (long double) layer_weight;
            if (obs.has_branch_counts && counts.hot > 0) {
                // Hot entries were useful under the current cache plan; keep a small
                // bias so layer reweighting does not churn the cache on every run.
                weighted += (long double) counts.hot * (long double) LLAMA_MOE_HOT_CACHE_HOT_STICKY_BONUS;
            }

            const uint64_t score = score_to_u64(weighted);
            if (score == 0) {
                continue;
            }
            result.push_back({ obs.layer, expert, score });
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

    size_t active_layers = 0;
    size_t total_hot_per_layer = 0;
    size_t min_hot_per_layer = std::numeric_limits<size_t>::max();
    size_t max_hot_per_layer = 0;

    for (const auto & experts : selected_by_layer) {
        if (experts.empty()) {
            continue;
        }

        active_layers++;
        total_hot_per_layer += experts.size();
        min_hot_per_layer = std::min(min_hot_per_layer, experts.size());
        max_hot_per_layer = std::max(max_hot_per_layer, experts.size());
    }

    LLAMA_LOG_INFO("%s: hot-cache active layers = %zu/%zu, hot experts per active layer min/avg/max = %zu/%.1f/%zu\n",
            __func__,
            active_layers,
            selected_by_layer.size(),
            active_layers > 0 ? min_hot_per_layer : 0,
            active_layers > 0 ? (double) total_hot_per_layer / (double) active_layers : 0.0,
            max_hot_per_layer);

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
        cache->layers[il].expert_weights_scale = model.hparams.expert_weights_scale;

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
        dst.hot_id_map_host.assign(n_expert, -1);

        for (uint32_t cache_id = 0; cache_id < experts.size(); ++cache_id) {
            const uint32_t expert = experts[cache_id];
            if (expert >= n_expert) {
                continue;
            }

            hot_id_map[expert] = int32_t(cache_id);
            hot_mask[expert] = 1.0f;
            cold_mask[expert] = 0.0f;
            dst.hot_id_map_host[expert] = int32_t(cache_id);

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

bool llama_moe_hot_cache_layer_active(const llama_model & model, int il) {
    return model.moe_hot_cache &&
           il >= 0 &&
           il < int(model.moe_hot_cache->layers.size()) &&
           model.moe_hot_cache->layers[il].active();
}

void llama_moe_hot_cache_build_worklist(
        ggml_tensor * dst,
        const ggml_tensor * selected_experts,
        const ggml_tensor * weights,
        const llama_moe_hot_cache_layer & layer,
        int ith,
        int nth) {
    GGML_UNUSED(nth);

    if (ith != 0) {
        return;
    }

    GGML_ASSERT(dst != nullptr);
    GGML_ASSERT(selected_experts != nullptr);
    GGML_ASSERT(weights != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(selected_experts->type == GGML_TYPE_I32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[1] == LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COUNT);
    GGML_ASSERT(weights->ne[0] == 1);
    GGML_ASSERT(selected_experts->ne[0] == weights->ne[1]);
    GGML_ASSERT(selected_experts->ne[1] == weights->ne[2]);
    GGML_ASSERT(int64_t(layer.hot_id_map_host.size()) == layer.n_expert);

    const int32_t capacity = dst->ne[0];
    const int32_t n_expert_used = selected_experts->ne[0];
    const int32_t n_tokens = selected_experts->ne[1];
    const int32_t total_slots = n_expert_used * n_tokens;
    const int32_t dummy_src_slot = total_slots;
    const float hot_padding_id =
        llama_moe_hot_cache_hot_dummy_padding() && layer.n_hot > 0 ? float(layer.n_hot) : -1.0f;

    GGML_ASSERT(capacity == total_slots);

    auto set_field = [&](int32_t field, int32_t slot, float value) {
        char * row = (char *) dst->data + field*dst->nb[1];
        *(float *)(row + slot*dst->nb[0]) = value;
    };

    for (int32_t slot = 0; slot < capacity; ++slot) {
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID,        slot, hot_padding_id);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT,  slot, float(dummy_src_slot));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID,  slot, 0.0f);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT,    slot, 0.0f);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID,       slot, -1.0f);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT, slot, float(dummy_src_slot));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, slot, 0.0f);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT,   slot, 0.0f);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, slot, -1.0f);
    }

    int32_t hot_slot = 0;
    int32_t cold_slot = 0;

    for (int32_t token = 0; token < n_tokens; ++token) {
        for (int32_t iex = 0; iex < n_expert_used; ++iex) {
            const int32_t expert = *(const int32_t *) ((const char *) selected_experts->data + iex*selected_experts->nb[0] + token*selected_experts->nb[1]);
            GGML_ASSERT(expert >= 0);
            GGML_ASSERT(expert < int32_t(layer.hot_id_map_host.size()));

            const float weight = *(const float *) ((const char *) weights->data + iex*weights->nb[1] + token*weights->nb[2]);
            const int32_t src_slot = token*n_expert_used + iex;
            const int32_t hot_id = layer.hot_id_map_host[expert];

            if (hot_id >= 0) {
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID,        hot_slot, float(hot_id));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT,  hot_slot, float(src_slot));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID,  hot_slot, float(token));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT,    hot_slot, weight);
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, hot_slot, float(expert));
                ++hot_slot;
            } else {
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID,       cold_slot, float(expert));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT, cold_slot, float(src_slot));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, cold_slot, float(token));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT,   cold_slot, weight);
                ++cold_slot;
            }
        }
    }

    GGML_ASSERT(hot_slot  <= capacity);
    GGML_ASSERT(cold_slot <= capacity);

    if (capacity > 0) {
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_COUNT,  0, float(hot_slot));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_COUNT, 0, float(cold_slot));
    }
}

void llama_moe_hot_cache_build_worklist_from_logits(
        ggml_tensor * dst,
        const ggml_tensor * logits,
        const llama_moe_hot_cache_layer & layer,
        int ith,
        int nth) {
    GGML_UNUSED(nth);

    if (ith != 0) {
        return;
    }

    GGML_ASSERT(dst != nullptr);
    GGML_ASSERT(logits != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[1] == LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COUNT);
    GGML_ASSERT(int64_t(layer.hot_id_map_host.size()) == layer.n_expert);
    GGML_ASSERT(logits->ne[0] == layer.n_expert);

    const int32_t capacity = dst->ne[0];
    const int32_t n_expert = logits->ne[0];
    const int32_t n_tokens = logits->ne[1];
    GGML_ASSERT(n_tokens > 0);
    GGML_ASSERT(capacity % n_tokens == 0);

    const int32_t n_expert_used = capacity / n_tokens;
    const int32_t total_slots = n_expert_used * n_tokens;
    const int32_t dummy_src_slot = total_slots;
    const float hot_padding_id =
        llama_moe_hot_cache_hot_dummy_padding() && layer.n_hot > 0 ? float(layer.n_hot) : -1.0f;
    GGML_ASSERT(capacity == total_slots);
    GGML_ASSERT(n_expert_used > 0);
    GGML_ASSERT(n_expert_used <= n_expert);
    GGML_ASSERT(n_expert_used <= LLAMA_MAX_EXPERTS);

    auto set_field = [&](int32_t field, int32_t slot, float value) {
        char * row = (char *) dst->data + field*dst->nb[1];
        *(float *)(row + slot*dst->nb[0]) = value;
    };

    for (int32_t slot = 0; slot < capacity; ++slot) {
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID,        slot, hot_padding_id);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT,  slot, float(dummy_src_slot));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID,  slot, 0.0f);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT,    slot, 0.0f);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID,       slot, -1.0f);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT, slot, float(dummy_src_slot));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, slot, 0.0f);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT,   slot, 0.0f);
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, slot, -1.0f);
    }

    int32_t hot_slot = 0;
    int32_t cold_slot = 0;
    int32_t top_experts[LLAMA_MAX_EXPERTS];
    float top_logits[LLAMA_MAX_EXPERTS];

    for (int32_t token = 0; token < n_tokens; ++token) {
        for (int32_t i = 0; i < n_expert_used; ++i) {
            top_experts[i] = -1;
            top_logits[i] = -std::numeric_limits<float>::infinity();
        }

        for (int32_t expert = 0; expert < n_expert; ++expert) {
            const float logit = *(const float *) ((const char *) logits->data + expert*logits->nb[0] + token*logits->nb[1]);

            for (int32_t pos = 0; pos < n_expert_used; ++pos) {
                if (logit <= top_logits[pos]) {
                    continue;
                }

                for (int32_t move = n_expert_used - 1; move > pos; --move) {
                    top_logits[move] = top_logits[move - 1];
                    top_experts[move] = top_experts[move - 1];
                }
                top_logits[pos] = logit;
                top_experts[pos] = expert;
                break;
            }
        }

        const float max_logit = top_logits[0];
        float weight_sum = 0.0f;
        for (int32_t iex = 0; iex < n_expert_used; ++iex) {
            top_logits[iex] = std::exp(top_logits[iex] - max_logit);
            weight_sum += top_logits[iex];
        }

        const float weight_scale =
            layer.expert_weights_scale != 0.0f && layer.expert_weights_scale != 1.0f ? layer.expert_weights_scale : 1.0f;

        for (int32_t iex = 0; iex < n_expert_used; ++iex) {
            const int32_t expert = top_experts[iex];
            GGML_ASSERT(expert >= 0);
            GGML_ASSERT(expert < int32_t(layer.hot_id_map_host.size()));

            const float weight = top_logits[iex] / weight_sum * weight_scale;
            const int32_t src_slot = token*n_expert_used + iex;
            const int32_t hot_id = layer.hot_id_map_host[expert];

            if (hot_id >= 0) {
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_ID,        hot_slot, float(hot_id));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_SRC_SLOT,  hot_slot, float(src_slot));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_TOKEN_ID,  hot_slot, float(token));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_WEIGHT,    hot_slot, weight);
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_EXPERT_ID, hot_slot, float(expert));
                ++hot_slot;
            } else {
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_ID,       cold_slot, float(expert));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_SRC_SLOT, cold_slot, float(src_slot));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_TOKEN_ID, cold_slot, float(token));
                set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_WEIGHT,   cold_slot, weight);
                ++cold_slot;
            }
        }
    }

    GGML_ASSERT(hot_slot  <= capacity);
    GGML_ASSERT(cold_slot <= capacity);

    if (capacity > 0) {
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_HOT_COUNT,  0, float(hot_slot));
        set_field(LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COLD_COUNT, 0, float(cold_slot));
    }
}
