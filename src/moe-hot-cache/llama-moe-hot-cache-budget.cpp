#include "llama-moe-hot-cache-budget.h"

#include "llama-moe-hot-cache-common.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

static void add_saturating(uint64_t & dst, uint64_t value) {
    if (dst > std::numeric_limits<uint64_t>::max() - value) {
        dst = std::numeric_limits<uint64_t>::max();
    } else {
        dst += value;
    }
}

static size_t mul_mib_saturating(size_t mib) {
    if (mib > std::numeric_limits<size_t>::max() / LLAMA_MOE_HOT_CACHE_MIB) {
        return std::numeric_limits<size_t>::max();
    }
    return mib * LLAMA_MOE_HOT_CACHE_MIB;
}

static size_t add_saturating_size(size_t lhs, size_t rhs) {
    if (lhs > std::numeric_limits<size_t>::max() - rhs) {
        return std::numeric_limits<size_t>::max();
    }
    return lhs + rhs;
}

static size_t estimate_kv_cache_bytes_on_device(
        const llama_model & model,
        const llama_model_params & params,
        ggml_backend_dev_t dev) {
    if (!params.moe_hot_cache_auto_offload_kqv) {
        return 0;
    }

    const auto & hparams = model.hparams;
    if (params.moe_hot_cache_auto_n_ctx == 0) {
        throw std::runtime_error("--moe-hot-cache-max-mib -1 requires an explicit --ctx-size");
    }

    const uint32_t n_seq_max = std::max<uint32_t>(1, params.moe_hot_cache_auto_n_seq_max);
    const uint32_t n_ubatch = std::max<uint32_t>(1, params.moe_hot_cache_auto_n_ubatch);
    uint32_t n_ctx = GGML_PAD(params.moe_hot_cache_auto_n_ctx, 256);
    uint32_t n_ctx_seq = n_ctx;
    if (!params.moe_hot_cache_auto_kv_unified) {
        n_ctx_seq = GGML_PAD(n_ctx / n_seq_max, 256);
        if (n_ctx_seq == 0) {
            throw std::runtime_error("--moe-hot-cache-max-mib -1 computed n_ctx_seq == 0");
        }
    }

    const uint32_t n_stream = params.moe_hot_cache_auto_kv_unified ? 1 : n_seq_max;
    const bool v_trans = params.moe_hot_cache_auto_flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED;
    const bool has_v = !hparams.is_mla();

    uint64_t result = 0;
    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        if (!hparams.has_kv(il) || model.dev_layer(il) != dev) {
            continue;
        }

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();
        uint32_t n_ctx_layer = n_ctx_seq;

        if (hparams.swa_type != LLAMA_SWA_TYPE_NONE && hparams.is_swa(il) && !params.moe_hot_cache_auto_swa_full) {
            n_ctx_layer = GGML_PAD(std::min(n_ctx_seq, hparams.n_swa*(params.moe_hot_cache_auto_kv_unified ? n_seq_max : 1) + n_ubatch), 256);
        }

        add_saturating(result, ggml_row_size(params.moe_hot_cache_auto_type_k, n_embd_k_gqa) * uint64_t(n_ctx_layer) * uint64_t(n_stream));
        if (has_v) {
            add_saturating(result, ggml_row_size(params.moe_hot_cache_auto_type_v, n_embd_v_gqa) * uint64_t(n_ctx_layer) * uint64_t(n_stream));
        }
    }

    return result > std::numeric_limits<size_t>::max() ? std::numeric_limits<size_t>::max() : size_t(result);
}

} // namespace

size_t llama_moe_hot_cache_compute_auto_budget_bytes(
        size_t free_bytes,
        size_t kv_reserve_bytes,
        size_t safety_reserve_mib) {
    const size_t safety_reserve = mul_mib_saturating(safety_reserve_mib);
    const size_t reserved = add_saturating_size(kv_reserve_bytes, safety_reserve);
    if (free_bytes <= reserved) {
        return 0;
    }

    return ((free_bytes - reserved) / LLAMA_MOE_HOT_CACHE_MIB) * LLAMA_MOE_HOT_CACHE_MIB;
}

ggml_backend_dev_t llama_moe_hot_cache_select_gpu_dev(const llama_model * model) {
    if (model != nullptr) {
        for (const auto & dev : model->devices) {
            const auto type = ggml_backend_dev_type(dev.dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                return dev.dev;
            }
        }
    }

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == nullptr) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    }
    if (dev == nullptr) {
        throw std::runtime_error("--moe-hot-cache requires a GPU backend device");
    }

    return dev;
}

size_t llama_moe_hot_cache_auto_budget_bytes(
        const llama_model & model,
        const llama_model_params & params,
        ggml_backend_dev_t dev,
        bool reserve_kv_cache) {
    size_t free = 0;
    size_t total = 0;
    ggml_backend_dev_memory(dev, &free, &total);
    GGML_UNUSED(total);

    const size_t kv_reserve = reserve_kv_cache ? estimate_kv_cache_bytes_on_device(model, params, dev) : 0;
    const size_t safety_reserve = mul_mib_saturating(params.moe_hot_cache_auto_reserve_mib);
    const size_t budget = llama_moe_hot_cache_compute_auto_budget_bytes(
            free,
            kv_reserve,
            params.moe_hot_cache_auto_reserve_mib);

    if (budget == 0) {
        LLAMA_LOG_WARN("%s: auto hot-cache budget on %s is 0 MiB: free before hot-cache = %zu MiB, %s KV reserve = %zu MiB, safety reserve = %zu MiB\n",
                __func__,
                ggml_backend_dev_name(dev),
                free / LLAMA_MOE_HOT_CACHE_MIB,
                reserve_kv_cache ? "estimated" : "deferred",
                kv_reserve / LLAMA_MOE_HOT_CACHE_MIB,
                safety_reserve / LLAMA_MOE_HOT_CACHE_MIB);
        return 0;
    }

    LLAMA_LOG_WARN("%s: auto hot-cache budget on %s: free before hot-cache = %zu MiB, %s KV reserve = %zu MiB, safety reserve = %zu MiB, budget = %zu MiB\n",
            __func__,
            ggml_backend_dev_name(dev),
            free / LLAMA_MOE_HOT_CACHE_MIB,
            reserve_kv_cache ? "estimated" : "deferred",
            kv_reserve / LLAMA_MOE_HOT_CACHE_MIB,
            safety_reserve / LLAMA_MOE_HOT_CACHE_MIB,
            budget / LLAMA_MOE_HOT_CACHE_MIB);

    return budget;
}
