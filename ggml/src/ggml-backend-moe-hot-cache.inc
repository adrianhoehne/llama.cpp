// Private MoE hot-cache scheduler extension. Included from ggml-backend.cpp so the
// scheduler internals stay file-private while the feature code remains isolated.

#include <condition_variable>
#include <mutex>
#include <thread>

struct ggml_backend_sched_moe_hot_cache_parallel_region_info {
    int32_t layer;
    int mode;
    int64_t max_slots;

    struct ggml_tensor * hot_count;
    struct ggml_tensor * cold_count;
    struct ggml_tensor * hot_start;
    struct ggml_tensor * hot_end;
    struct ggml_tensor * hot_output;
    struct ggml_tensor * cold_start;
    struct ggml_tensor * cold_end;
    struct ggml_tensor * cold_output;
    struct ggml_tensor * join;
    uint32_t flags;

    int hot_split_begin;
    int hot_split_end;
    int cold_split_begin;
    int cold_split_end;
    int join_split;
};

struct ggml_backend_sched_moe_parallel_worker;

struct ggml_backend_sched_moe_hot_cache_parallel {
    std::vector<ggml_backend_sched_moe_hot_cache_parallel_region_info> regions;
    std::vector<ggml_backend_sched_moe_hot_cache_parallel_perf> perf;
    ggml_backend_sched_moe_parallel_worker * worker = nullptr;
    bool perf_enabled = true;
};

static void ggml_backend_sched_moe_parallel_worker_free(ggml_backend_sched_moe_parallel_worker * worker);

static void ggml_backend_sched_moe_hot_cache_parallel_init(ggml_backend_sched_t sched) {
    sched->moe_hot_cache_parallel = new ggml_backend_sched_moe_hot_cache_parallel();
}

static void ggml_backend_sched_moe_hot_cache_parallel_free(ggml_backend_sched_t sched) {
    if (sched->moe_hot_cache_parallel == nullptr) {
        return;
    }

    ggml_backend_sched_moe_parallel_worker_free(sched->moe_hot_cache_parallel->worker);
    delete sched->moe_hot_cache_parallel;
    sched->moe_hot_cache_parallel = nullptr;
}

static void ggml_backend_sched_moe_hot_cache_parallel_reset(ggml_backend_sched_t sched) {
    if (sched->moe_hot_cache_parallel != nullptr) {
        sched->moe_hot_cache_parallel->regions.clear();
    }
}

static bool ggml_backend_sched_moe_hot_cache_parallel_is_split_boundary(
        ggml_backend_sched_t sched,
        const struct ggml_tensor * node) {
    if (sched->moe_hot_cache_parallel == nullptr) {
        return false;
    }

    for (const auto & region : sched->moe_hot_cache_parallel->regions) {
        if (node == region.hot_start || node == region.cold_start || node == region.join) {
            return true;
        }
    }

    return false;
}

struct ggml_backend_sched_moe_parallel_worker {
    std::mutex mutex;
    std::condition_variable cv;
    std::thread thread;

    bool stop = false;
    bool has_job = false;
    bool done = true;

    ggml_backend_sched_t sched = nullptr;
    int split_begin = 0;
    int split_end = -1;
    ggml_backend_sched_compute_state * state = nullptr;

    enum ggml_status status = GGML_STATUS_SUCCESS;
    int64_t end_us = 0;
};

static void ggml_backend_sched_moe_parallel_worker_loop(ggml_backend_sched_moe_parallel_worker * worker) {
    for (;;) {
        ggml_backend_sched_t sched = nullptr;
        int split_begin = 0;
        int split_end = -1;
        ggml_backend_sched_compute_state * state = nullptr;

        {
            std::unique_lock<std::mutex> lock(worker->mutex);
            worker->cv.wait(lock, [worker]() {
                return worker->stop || worker->has_job;
            });

            if (worker->stop) {
                return;
            }

            sched = worker->sched;
            split_begin = worker->split_begin;
            split_end = worker->split_end;
            state = worker->state;
            worker->has_job = false;
        }

        enum ggml_status status = ggml_backend_sched_compute_split_range(
                sched,
                split_begin,
                split_end,
                *state,
                /* use_callback = */ false,
                /* first_inputs_prepared = */ true);
        const int64_t end_us = sched->moe_hot_cache_parallel->perf_enabled ? ggml_time_us() : 0;

        {
            std::lock_guard<std::mutex> lock(worker->mutex);
            worker->status = status;
            worker->end_us = end_us;
            worker->done = true;
        }
        worker->cv.notify_all();
    }
}

static ggml_backend_sched_moe_parallel_worker * ggml_backend_sched_moe_parallel_worker_new() {
    auto * worker = new ggml_backend_sched_moe_parallel_worker();
    worker->thread = std::thread(ggml_backend_sched_moe_parallel_worker_loop, worker);
    return worker;
}

static void ggml_backend_sched_moe_parallel_worker_free(ggml_backend_sched_moe_parallel_worker * worker) {
    if (worker == nullptr) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(worker->mutex);
        worker->stop = true;
    }
    worker->cv.notify_all();

    if (worker->thread.joinable()) {
        worker->thread.join();
    }

    delete worker;
}

static ggml_backend_sched_moe_parallel_worker * ggml_backend_sched_moe_parallel_worker_get(ggml_backend_sched_t sched) {
    if (sched->moe_hot_cache_parallel->worker == nullptr) {
        sched->moe_hot_cache_parallel->worker = ggml_backend_sched_moe_parallel_worker_new();
    }
    return sched->moe_hot_cache_parallel->worker;
}

static void ggml_backend_sched_moe_parallel_worker_start(
        ggml_backend_sched_t sched,
        int split_begin,
        int split_end,
        ggml_backend_sched_compute_state & state) {
    ggml_backend_sched_moe_parallel_worker * worker = ggml_backend_sched_moe_parallel_worker_get(sched);

    {
        std::lock_guard<std::mutex> lock(worker->mutex);
        GGML_ASSERT(worker->done);
        GGML_ASSERT(!worker->has_job);
        worker->sched = sched;
        worker->split_begin = split_begin;
        worker->split_end = split_end;
        worker->state = &state;
        worker->status = GGML_STATUS_SUCCESS;
        worker->end_us = 0;
        worker->done = false;
        worker->has_job = true;
    }
    worker->cv.notify_one();
}

static enum ggml_status ggml_backend_sched_moe_parallel_worker_wait(
        ggml_backend_sched_t sched,
        int64_t & end_us) {
    ggml_backend_sched_moe_parallel_worker * worker = sched->moe_hot_cache_parallel->worker;
    GGML_ASSERT(worker != nullptr);

    std::unique_lock<std::mutex> lock(worker->mutex);
    worker->cv.wait(lock, [worker]() {
        return worker->done;
    });

    end_us = worker->end_us;
    return worker->status;
}

static int ggml_backend_sched_find_split_containing(
        ggml_backend_sched_t sched,
        const struct ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return -1;
    }

    for (int split_id = 0; split_id < sched->n_splits; ++split_id) {
        const ggml_cgraph & graph = sched->splits[split_id].graph;
        for (int node_id = 0; node_id < graph.n_nodes; ++node_id) {
            if (graph.nodes[node_id] == tensor) {
                return split_id;
            }
        }
    }

    return -1;
}

static int32_t ggml_backend_sched_moe_parallel_split_backend_id(
        ggml_backend_sched_t sched,
        int split_id) {
    if (split_id < 0 || split_id >= sched->n_splits) {
        return -1;
    }

    return sched->splits[split_id].backend_id;
}

static void ggml_backend_sched_moe_parallel_fill_split_debug(
        ggml_backend_sched_t sched,
        const ggml_backend_sched_moe_hot_cache_parallel_region_info & region,
        ggml_backend_sched_moe_hot_cache_parallel_perf & perf) {
    const int hot_count_split  = ggml_backend_sched_find_split_containing(sched, region.hot_count);
    const int cold_count_split = ggml_backend_sched_find_split_containing(sched, region.cold_count);

    perf.parallel_split_debug_samples = 1;
    perf.parallel_split_debug_hot_begin = region.hot_split_begin;
    perf.parallel_split_debug_hot_end = region.hot_split_end;
    perf.parallel_split_debug_cold_begin = region.cold_split_begin;
    perf.parallel_split_debug_cold_end = region.cold_split_end;
    perf.parallel_split_debug_join = region.join_split;
    perf.parallel_split_debug_hot_count = hot_count_split;
    perf.parallel_split_debug_cold_count = cold_count_split;
    perf.parallel_split_debug_hot_backend = ggml_backend_sched_moe_parallel_split_backend_id(sched, region.hot_split_begin);
    perf.parallel_split_debug_cold_backend = ggml_backend_sched_moe_parallel_split_backend_id(sched, region.cold_split_begin);
    perf.parallel_split_debug_join_backend = ggml_backend_sched_moe_parallel_split_backend_id(sched, region.join_split);
}

static bool ggml_backend_sched_backend_is_cuda(ggml_backend_t backend) {
    const char * name = ggml_backend_name(backend);
    return name != nullptr && strstr(name, "CUDA") != nullptr;
}

static bool ggml_backend_sched_backend_is_cpu(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    return dev != nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

static void ggml_backend_sched_moe_parallel_record_fallback(
        ggml_backend_sched_t sched,
        const ggml_backend_sched_moe_hot_cache_parallel_region_info & region,
        const char * reason) {
    if (!sched->moe_hot_cache_parallel->perf_enabled) {
        return;
    }

    ggml_backend_sched_moe_hot_cache_parallel_perf perf = {};
    perf.layer = region.layer;
    perf.parallel_fallbacks = 1;
    ggml_backend_sched_moe_parallel_fill_split_debug(sched, region, perf);

    if (reason != nullptr && strcmp(reason, "incomplete region annotation") == 0) {
        perf.parallel_fallback_incomplete = 1;
    } else if (reason != nullptr && strcmp(reason, "count tensors are not in the serial prefix") == 0) {
        perf.parallel_fallback_count_not_prefix = 1;
    } else if (reason != nullptr && strcmp(reason, "region split order does not contain two contiguous lanes before join") == 0) {
        perf.parallel_fallback_bad_split_order = 1;
    } else if (reason != nullptr && strcmp(reason, "hot and cold lanes resolved to the same backend") == 0) {
        perf.parallel_fallback_same_backend = 1;
    } else if (reason != nullptr && strcmp(reason, "hot lane spans multiple backends") == 0) {
        perf.parallel_fallback_hot_spans_backends = 1;
    } else if (reason != nullptr && strcmp(reason, "cold lane spans multiple backends") == 0) {
        perf.parallel_fallback_cold_spans_backends = 1;
    } else if (reason != nullptr && strcmp(reason, "hot lane is not on a CUDA backend") == 0) {
        perf.parallel_fallback_hot_not_cuda = 1;
    } else if (reason != nullptr && strcmp(reason, "cold lane is not on a CPU backend") == 0) {
        perf.parallel_fallback_cold_not_cpu = 1;
    } else if (reason != nullptr && strcmp(reason, "count readback failed") == 0) {
        perf.parallel_fallback_count_readback = 1;
    } else if (reason != nullptr && strcmp(reason, "parallel region below auto slot threshold") == 0) {
        perf.parallel_fallback_threshold = 1;
    } else if (reason != nullptr &&
            (strcmp(reason, "failed to zero skipped hot lane") == 0 ||
             strcmp(reason, "failed to zero skipped cold lane") == 0)) {
        perf.parallel_fallback_zero_output = 1;
    } else {
        perf.parallel_fallback_other = 1;
    }

    sched->moe_hot_cache_parallel->perf.push_back(perf);
}

static bool ggml_backend_sched_moe_parallel_fail_or_fallback(
        ggml_backend_sched_t sched,
        const ggml_backend_sched_moe_hot_cache_parallel_region_info & region,
        const char * reason) {
    if (region.mode == 2) {
        GGML_LOG_ERROR("%s: forced MoE hot-cache parallel region failed for layer %d: %s "
                "(hot=%d..%d cold=%d..%d join=%d)\n",
                __func__, region.layer, reason,
                region.hot_split_begin, region.hot_split_end,
                region.cold_split_begin, region.cold_split_end,
                region.join_split);
        return false;
    }

    static std::mutex log_mutex;
    static bool logged = false;
    {
        std::lock_guard<std::mutex> lock(log_mutex);
        if (!logged) {
            GGML_LOG_INFO("%s: MoE hot-cache parallel mode fell back to serial execution: %s\n", __func__, reason);
            logged = true;
        }
    }

    ggml_backend_sched_moe_parallel_record_fallback(sched, region, reason);
    return true;
}

static bool ggml_backend_sched_resolve_moe_parallel_region(
        ggml_backend_sched_t sched,
        ggml_backend_sched_moe_hot_cache_parallel_region_info & region,
        const char ** reason) {
    region.hot_split_begin = ggml_backend_sched_find_split_containing(sched, region.hot_start);
    region.hot_split_end   = ggml_backend_sched_find_split_containing(sched, region.hot_end);
    region.cold_split_begin = ggml_backend_sched_find_split_containing(sched, region.cold_start);
    region.cold_split_end   = ggml_backend_sched_find_split_containing(sched, region.cold_end);
    region.join_split       = ggml_backend_sched_find_split_containing(sched, region.join);

    const int hot_count_split = ggml_backend_sched_find_split_containing(sched, region.hot_count);
    const int cold_count_split = ggml_backend_sched_find_split_containing(sched, region.cold_count);

    if (region.hot_split_begin < 0 || region.hot_split_end < 0 ||
        region.cold_split_begin < 0 || region.cold_split_end < 0 ||
        region.join_split < 0 || hot_count_split < 0 || cold_count_split < 0) {
        *reason = "incomplete region annotation";
        return false;
    }

    const int first_lane_split = std::min(region.hot_split_begin, region.cold_split_begin);
    if (!(hot_count_split < first_lane_split && cold_count_split < first_lane_split)) {
        *reason = "count tensors are not in the serial prefix";
        return false;
    }

    const bool hot_then_cold_lanes =
        region.hot_split_begin <= region.hot_split_end &&
        region.hot_split_end < region.cold_split_begin &&
        region.cold_split_begin <= region.cold_split_end &&
        region.cold_split_end < region.join_split &&
        region.hot_split_end + 1 == region.cold_split_begin;

    const bool cold_then_hot_lanes =
        region.cold_split_begin <= region.cold_split_end &&
        region.cold_split_end < region.hot_split_begin &&
        region.hot_split_begin <= region.hot_split_end &&
        region.hot_split_end < region.join_split &&
        region.cold_split_end + 1 == region.hot_split_begin;

    const int second_lane_end = hot_then_cold_lanes ? region.cold_split_end :
        cold_then_hot_lanes ? region.hot_split_end : -1;
    const bool allow_join_bridge =
        (region.flags & GGML_BACKEND_SCHED_MOE_HOT_CACHE_PARALLEL_FLAG_ALLOW_JOIN_BRIDGE) != 0;
    const int join_gap = second_lane_end >= 0 ? region.join_split - second_lane_end : -1;
    const bool join_is_contiguous = join_gap == 1;
    const bool join_has_single_bridge = allow_join_bridge && join_gap == 2;

    if ((!hot_then_cold_lanes && !cold_then_hot_lanes) || (!join_is_contiguous && !join_has_single_bridge)) {
        *reason = "region split order does not contain two contiguous lanes before join";
        return false;
    }

    const int hot_backend_id = sched->splits[region.hot_split_begin].backend_id;
    const int cold_backend_id = sched->splits[region.cold_split_begin].backend_id;

    if (hot_backend_id == cold_backend_id) {
        *reason = "hot and cold lanes resolved to the same backend";
        return false;
    }

    for (int split_id = region.hot_split_begin; split_id <= region.hot_split_end; ++split_id) {
        if (sched->splits[split_id].backend_id != hot_backend_id) {
            *reason = "hot lane spans multiple backends";
            return false;
        }
    }

    for (int split_id = region.cold_split_begin; split_id <= region.cold_split_end; ++split_id) {
        if (sched->splits[split_id].backend_id != cold_backend_id) {
            *reason = "cold lane spans multiple backends";
            return false;
        }
    }

    if (!ggml_backend_sched_backend_is_cuda(sched->backends[hot_backend_id])) {
        *reason = "hot lane is not on a CUDA backend";
        return false;
    }

    if (!ggml_backend_sched_backend_is_cpu(sched->backends[cold_backend_id])) {
        *reason = "cold lane is not on a CPU backend";
        return false;
    }

    return true;
}

static int64_t ggml_backend_sched_read_f32_count(
        ggml_backend_sched_t sched,
        struct ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32) {
        return -1;
    }

    ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
    if (backend == nullptr) {
        return -1;
    }

    float value = 0.0f;
    if (ggml_backend_sched_backend_is_cpu(backend)) {
        if (tensor->data == nullptr) {
            return -1;
        }
        value = *(const float *) tensor->data;
    } else {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, &value, 0, sizeof(value));
    }

    if (value <= 0.0f) {
        return 0;
    }

    return (int64_t) (value + 0.5f);
}

static int64_t ggml_backend_sched_moe_parallel_auto_min_slots() {
    static const int64_t value = []() {
        const char * env = getenv("LLAMA_MOE_HOT_CACHE_PARALLEL_MIN_SLOTS");
        if (env == nullptr || env[0] == '\0') {
            return int64_t(64);
        }

        char * end = nullptr;
        const long long parsed = strtoll(env, &end, 10);
        if (end == env || parsed < 0) {
            return int64_t(64);
        }

        return (int64_t) parsed;
    }();

    return value;
}

static bool ggml_backend_sched_zero_tensor(struct ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->buffer == nullptr || !ggml_is_contiguous(tensor)) {
        return false;
    }

    ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    return true;
}

static enum ggml_status ggml_backend_sched_compute_moe_parallel_region(
        ggml_backend_sched_t sched,
        const ggml_backend_sched_moe_hot_cache_parallel_region_info & region,
        bool & serial_fallback) {
    serial_fallback = false;

    if (region.mode != 2) {
        const int64_t min_slots = ggml_backend_sched_moe_parallel_auto_min_slots();
        if (min_slots > 0 && region.max_slots >= 0 && region.max_slots < min_slots) {
            if (!ggml_backend_sched_moe_parallel_fail_or_fallback(sched, region, "parallel region below auto slot threshold")) {
                return GGML_STATUS_FAILED;
            }
            serial_fallback = true;
            return GGML_STATUS_SUCCESS;
        }
    }

    const int64_t hot_count = ggml_backend_sched_read_f32_count(sched, region.hot_count);
    const int64_t cold_count = ggml_backend_sched_read_f32_count(sched, region.cold_count);

    if (hot_count < 0 || cold_count < 0) {
        if (!ggml_backend_sched_moe_parallel_fail_or_fallback(sched, region, "count readback failed")) {
            return GGML_STATUS_FAILED;
        }
        serial_fallback = true;
        return GGML_STATUS_SUCCESS;
    }

    if (region.mode != 2 && hot_count > 0 && cold_count > 0) {
        const int64_t min_slots = ggml_backend_sched_moe_parallel_auto_min_slots();
        if (hot_count + cold_count < min_slots) {
            if (!ggml_backend_sched_moe_parallel_fail_or_fallback(sched, region, "parallel region below auto slot threshold")) {
                return GGML_STATUS_FAILED;
            }
            serial_fallback = true;
            return GGML_STATUS_SUCCESS;
        }
    }

    const bool collect_perf = sched->moe_hot_cache_parallel->perf_enabled;
    ggml_backend_sched_moe_hot_cache_parallel_perf perf = {};
    perf.layer = region.layer;
    if (collect_perf) {
        perf.parallel_hot_count = hot_count > 0 ? (uint64_t) hot_count : 0;
        perf.parallel_cold_count = cold_count > 0 ? (uint64_t) cold_count : 0;
        ggml_backend_sched_moe_parallel_fill_split_debug(sched, region, perf);
    }

    const int64_t region_start_us = collect_perf ? ggml_time_us() : 0;

    ggml_backend_sched_compute_state hot_state;
    ggml_backend_sched_compute_state cold_state;

    enum ggml_status hot_status = GGML_STATUS_SUCCESS;
    bool cold_worker_started = false;

    int64_t hot_lane_start_us = 0;
    int64_t hot_lane_end_us = 0;
    int64_t cold_lane_start_us = 0;
    int64_t cold_lane_end_us = 0;

    if (cold_count > 0) {
        if (collect_perf) {
            perf.parallel_cold_launches = 1;
            cold_lane_start_us = ggml_time_us();
        }

        enum ggml_status ec = ggml_backend_sched_compute_split_inputs(
                sched, &sched->splits[region.cold_split_begin], cold_state);
        if (ec != GGML_STATUS_SUCCESS) {
            return ec;
        }

        ggml_backend_sched_moe_parallel_worker_start(
                sched,
                region.cold_split_begin,
                region.cold_split_end,
                cold_state);
        cold_worker_started = true;
    } else if (collect_perf) {
        perf.parallel_cold_skips_zero = 1;
    }

    if (hot_count > 0) {
        if (collect_perf) {
            perf.parallel_hot_launches = 1;
            hot_lane_start_us = ggml_time_us();
        }

        enum ggml_status ec = ggml_backend_sched_compute_split_inputs(
                sched, &sched->splits[region.hot_split_begin], hot_state);
        if (ec != GGML_STATUS_SUCCESS) {
            if (cold_worker_started) {
                int64_t ignored_end_us = 0;
                ggml_backend_sched_moe_parallel_worker_wait(sched, ignored_end_us);
            }
            return ec;
        }
    } else if (collect_perf) {
        perf.parallel_hot_skips_zero = 1;
    }

    if (hot_count > 0) {
        hot_status = ggml_backend_sched_compute_split_range(
                sched,
                region.hot_split_begin,
                region.hot_split_end,
                hot_state,
                /* use_callback = */ false,
                /* first_inputs_prepared = */ true);
        if (hot_status != GGML_STATUS_SUCCESS) {
            if (cold_worker_started) {
                int64_t ignored_end_us = 0;
                ggml_backend_sched_moe_parallel_worker_wait(sched, ignored_end_us);
            }
            return hot_status;
        }
    }

    const int64_t join_wait_start_us = collect_perf ? ggml_time_us() : 0;

    if (hot_count > 0) {
        ggml_backend_synchronize(sched->backends[sched->splits[region.hot_split_begin].backend_id]);
        if (collect_perf) {
            hot_lane_end_us = ggml_time_us();
        }
    }

    if (cold_worker_started) {
        int64_t worker_end_us = 0;
        const enum ggml_status ec = ggml_backend_sched_moe_parallel_worker_wait(sched, worker_end_us);
        if (collect_perf) {
            cold_lane_end_us = worker_end_us;
        }
        if (ec != GGML_STATUS_SUCCESS) {
            return ec;
        }
    }

    if (cold_count > 0) {
        ggml_backend_synchronize(sched->backends[sched->splits[region.cold_split_begin].backend_id]);
        if (collect_perf && cold_lane_end_us == 0) {
            cold_lane_end_us = ggml_time_us();
        }
    }

    const int64_t join_wait_end_us = collect_perf ? ggml_time_us() : 0;

    if (hot_count == 0 && !ggml_backend_sched_zero_tensor(region.hot_output)) {
        if (!ggml_backend_sched_moe_parallel_fail_or_fallback(sched, region, "failed to zero skipped hot lane")) {
            return GGML_STATUS_FAILED;
        }
        serial_fallback = true;
        return GGML_STATUS_SUCCESS;
    }

    if (cold_count == 0 && !ggml_backend_sched_zero_tensor(region.cold_output)) {
        if (!ggml_backend_sched_moe_parallel_fail_or_fallback(sched, region, "failed to zero skipped cold lane")) {
            return GGML_STATUS_FAILED;
        }
        serial_fallback = true;
        return GGML_STATUS_SUCCESS;
    }

    if (collect_perf) {
        const int64_t region_end_us = ggml_time_us();

        perf.parallel_region_wall_time_us = region_end_us > region_start_us ? (uint64_t) (region_end_us - region_start_us) : 0;
        perf.parallel_hot_lane_wall_time_us =
            hot_lane_start_us > 0 && hot_lane_end_us > hot_lane_start_us ? (uint64_t) (hot_lane_end_us - hot_lane_start_us) : 0;
        perf.parallel_cold_lane_wall_time_us =
            cold_lane_start_us > 0 && cold_lane_end_us > cold_lane_start_us ?
                (uint64_t) (cold_lane_end_us - cold_lane_start_us) : 0;
        perf.parallel_join_wait_time_us = join_wait_end_us > join_wait_start_us ? (uint64_t) (join_wait_end_us - join_wait_start_us) : 0;

        const uint64_t lane_sum = perf.parallel_hot_lane_wall_time_us + perf.parallel_cold_lane_wall_time_us;
        perf.parallel_overlap_estimate_us = lane_sum > perf.parallel_region_wall_time_us ? lane_sum - perf.parallel_region_wall_time_us : 0;

        sched->moe_hot_cache_parallel->perf.push_back(perf);
    }

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);

    sched->moe_hot_cache_parallel->perf.clear();

    ggml_backend_sched_compute_state serial_state;
    int split_id = 0;

    for (auto region : sched->moe_hot_cache_parallel->regions) {
        const char * reason = nullptr;
        if (!ggml_backend_sched_resolve_moe_parallel_region(sched, region, &reason)) {
            if (!ggml_backend_sched_moe_parallel_fail_or_fallback(sched, region, reason)) {
                return GGML_STATUS_FAILED;
            }
            continue;
        }

        const int region_split_begin = std::min(region.hot_split_begin, region.cold_split_begin);

        if (split_id > region_split_begin) {
            if (!ggml_backend_sched_moe_parallel_fail_or_fallback(sched, region, "parallel region overlaps an already executed split")) {
                return GGML_STATUS_FAILED;
            }
            continue;
        }

        while (split_id < region_split_begin) {
            enum ggml_status ec = ggml_backend_sched_compute_split(sched, split_id, serial_state, /* use_callback = */ true);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
            split_id++;
        }

        bool serial_fallback = false;
        enum ggml_status ec = ggml_backend_sched_compute_moe_parallel_region(sched, region, serial_fallback);
        if (serial_fallback) {
            while (split_id < region.join_split) {
                ec = ggml_backend_sched_compute_split(sched, split_id, serial_state, /* use_callback = */ true);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }
                split_id++;
            }
        } else if (ec != GGML_STATUS_SUCCESS) {
            return ec;
        } else {
            split_id = std::max(region.hot_split_end, region.cold_split_end) + 1;
        }
    }

    while (split_id < sched->n_splits) {
        enum ggml_status ec = ggml_backend_sched_compute_split(sched, split_id, serial_state, /* use_callback = */ true);
        if (ec != GGML_STATUS_SUCCESS) {
            return ec;
        }
        split_id++;
    }

    return GGML_STATUS_SUCCESS;
}

void ggml_backend_sched_set_moe_hot_cache_parallel_perf_enabled(ggml_backend_sched_t sched, bool enabled) {
    GGML_ASSERT(sched);
    sched->moe_hot_cache_parallel->perf_enabled = enabled;
}

void ggml_backend_sched_moe_hot_cache_parallel_region(
        ggml_backend_sched_t sched,
        int32_t layer,
        int mode,
        int64_t max_slots,
        struct ggml_tensor * hot_count,
        struct ggml_tensor * cold_count,
        struct ggml_tensor * hot_start,
        struct ggml_tensor * hot_end,
        struct ggml_tensor * hot_output,
        struct ggml_tensor * cold_start,
        struct ggml_tensor * cold_end,
        struct ggml_tensor * cold_output,
        struct ggml_tensor * join,
        uint32_t flags) {
    GGML_ASSERT(sched);

    if (mode == 0) {
        return;
    }

    ggml_backend_sched_moe_hot_cache_parallel_region_info region = {};
    region.layer = layer;
    region.mode = mode;
    region.max_slots = max_slots;
    region.hot_count = hot_count;
    region.cold_count = cold_count;
    region.hot_start = hot_start;
    region.hot_end = hot_end;
    region.hot_output = hot_output;
    region.cold_start = cold_start;
    region.cold_end = cold_end;
    region.cold_output = cold_output;
    region.join = join;
    region.flags = flags;
    region.hot_split_begin = -1;
    region.hot_split_end = -1;
    region.cold_split_begin = -1;
    region.cold_split_end = -1;
    region.join_split = -1;

    sched->moe_hot_cache_parallel->regions.push_back(region);
}

int ggml_backend_sched_get_moe_hot_cache_parallel_perf(
        ggml_backend_sched_t sched,
        struct ggml_backend_sched_moe_hot_cache_parallel_perf * out,
        int max_entries) {
    GGML_ASSERT(sched);

    const int n = (int) sched->moe_hot_cache_parallel->perf.size();
    if (out == nullptr || max_entries <= 0) {
        return n;
    }

    const int n_copy = std::min(n, max_entries);
    for (int i = 0; i < n_copy; ++i) {
        out[i] = (sched->moe_hot_cache_parallel->perf)[i];
    }

    return n;
}
