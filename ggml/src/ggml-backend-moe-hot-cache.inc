// Private MoE hot-cache scheduler extension. Included from ggml-backend.cpp so the
// scheduler internals stay file-private while the feature code remains isolated.

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>

struct ggml_backend_sched_moe_hot_cache_parallel_lane_info {
    int32_t index;
    bool hot;

    struct ggml_tensor * count;
    struct ggml_tensor * start;
    struct ggml_tensor * end;
    struct ggml_tensor * output;

    int split_begin;
    int split_end;
};

struct ggml_backend_sched_moe_hot_cache_parallel_region_info {
    int32_t layer;
    int mode;
    int64_t max_slots;

    std::vector<ggml_backend_sched_moe_hot_cache_parallel_lane_info> hot_lanes;
    ggml_backend_sched_moe_hot_cache_parallel_lane_info cold_lane;
    bool has_cold_lane;

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
    std::vector<ggml_backend_sched_moe_parallel_worker *> workers;
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

    for (ggml_backend_sched_moe_parallel_worker * worker : sched->moe_hot_cache_parallel->workers) {
        ggml_backend_sched_moe_parallel_worker_free(worker);
    }
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
        for (const auto & lane : region.hot_lanes) {
            if (node == lane.start) {
                return true;
            }
        }
        if (region.has_cold_lane && node == region.cold_lane.start) {
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

static ggml_backend_sched_moe_parallel_worker * ggml_backend_sched_moe_parallel_worker_get(
        ggml_backend_sched_t sched,
        size_t worker_index) {
    auto & workers = sched->moe_hot_cache_parallel->workers;
    while (workers.size() <= worker_index) {
        workers.push_back(ggml_backend_sched_moe_parallel_worker_new());
    }
    return workers[worker_index];
}

static void ggml_backend_sched_moe_parallel_worker_start(
        ggml_backend_sched_t sched,
        size_t worker_index,
        int split_begin,
        int split_end,
        ggml_backend_sched_compute_state & state) {
    ggml_backend_sched_moe_parallel_worker * worker = ggml_backend_sched_moe_parallel_worker_get(sched, worker_index);

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
        size_t worker_index,
        int64_t & end_us) {
    GGML_ASSERT(worker_index < sched->moe_hot_cache_parallel->workers.size());
    ggml_backend_sched_moe_parallel_worker * worker = sched->moe_hot_cache_parallel->workers[worker_index];
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
    int hot_split_begin = -1;
    int hot_split_end = -1;
    int hot_count_split = -1;

    for (const auto & lane : region.hot_lanes) {
        if (lane.split_begin < 0 || lane.split_end < 0) {
            continue;
        }
        hot_split_begin = hot_split_begin < 0 ? lane.split_begin : std::min(hot_split_begin, lane.split_begin);
        hot_split_end = std::max(hot_split_end, lane.split_end);
        if (hot_count_split < 0) {
            hot_count_split = ggml_backend_sched_find_split_containing(sched, lane.count);
        }
    }

    if (hot_split_begin < 0) {
        hot_split_begin = region.hot_split_begin;
        hot_split_end = region.hot_split_end;
        hot_count_split = ggml_backend_sched_find_split_containing(sched, region.hot_count);
    }

    const int cold_count_split = region.has_cold_lane
        ? ggml_backend_sched_find_split_containing(sched, region.cold_lane.count)
        : ggml_backend_sched_find_split_containing(sched, region.cold_count);

    perf.parallel_split_debug_samples = 1;
    perf.parallel_split_debug_hot_begin = hot_split_begin;
    perf.parallel_split_debug_hot_end = hot_split_end;
    perf.parallel_split_debug_cold_begin = region.has_cold_lane ? region.cold_lane.split_begin : region.cold_split_begin;
    perf.parallel_split_debug_cold_end = region.has_cold_lane ? region.cold_lane.split_end : region.cold_split_end;
    perf.parallel_split_debug_join = region.join_split;
    perf.parallel_split_debug_hot_count = hot_count_split;
    perf.parallel_split_debug_cold_count = cold_count_split;
    perf.parallel_split_debug_hot_backend = ggml_backend_sched_moe_parallel_split_backend_id(sched, hot_split_begin);
    perf.parallel_split_debug_cold_backend = ggml_backend_sched_moe_parallel_split_backend_id(
            sched,
            region.has_cold_lane ? region.cold_lane.split_begin : region.cold_split_begin);
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
    if (region.hot_lanes.empty() && region.hot_start != nullptr) {
        ggml_backend_sched_moe_hot_cache_parallel_lane_info lane = {};
        lane.index = 0;
        lane.hot = true;
        lane.count = region.hot_count;
        lane.start = region.hot_start;
        lane.end = region.hot_end;
        lane.output = region.hot_output;
        lane.split_begin = -1;
        lane.split_end = -1;
        region.hot_lanes.push_back(lane);
    }

    if (!region.has_cold_lane && region.cold_start != nullptr) {
        region.cold_lane.index = -1;
        region.cold_lane.hot = false;
        region.cold_lane.count = region.cold_count;
        region.cold_lane.start = region.cold_start;
        region.cold_lane.end = region.cold_end;
        region.cold_lane.output = region.cold_output;
        region.cold_lane.split_begin = -1;
        region.cold_lane.split_end = -1;
        region.has_cold_lane = true;
    }

    region.join_split = ggml_backend_sched_find_split_containing(sched, region.join);

    if (region.hot_lanes.empty() || region.join_split < 0) {
        *reason = "incomplete region annotation";
        return false;
    }

    int first_lane_split = -1;
    int last_lane_split = -1;

    for (auto & lane : region.hot_lanes) {
        lane.split_begin = ggml_backend_sched_find_split_containing(sched, lane.start);
        lane.split_end = ggml_backend_sched_find_split_containing(sched, lane.end);
        const int count_split = ggml_backend_sched_find_split_containing(sched, lane.count);
        if (lane.split_begin < 0 || lane.split_end < 0 || count_split < 0) {
            *reason = "incomplete region annotation";
            return false;
        }
        first_lane_split = first_lane_split < 0 ? lane.split_begin : std::min(first_lane_split, lane.split_begin);
        last_lane_split = std::max(last_lane_split, lane.split_end);
    }

    if (region.has_cold_lane) {
        region.cold_lane.split_begin = ggml_backend_sched_find_split_containing(sched, region.cold_lane.start);
        region.cold_lane.split_end = ggml_backend_sched_find_split_containing(sched, region.cold_lane.end);
        const int cold_count_split = ggml_backend_sched_find_split_containing(sched, region.cold_lane.count);
        if (region.cold_lane.split_begin < 0 || region.cold_lane.split_end < 0 || cold_count_split < 0) {
            *reason = "incomplete region annotation";
            return false;
        }
        first_lane_split = first_lane_split < 0 ? region.cold_lane.split_begin : std::min(first_lane_split, region.cold_lane.split_begin);
        last_lane_split = std::max(last_lane_split, region.cold_lane.split_end);
    }

    for (const auto & lane : region.hot_lanes) {
        const int count_split = ggml_backend_sched_find_split_containing(sched, lane.count);
        if (count_split >= first_lane_split) {
            *reason = "count tensors are not in the serial prefix";
            return false;
        }
    }

    if (region.has_cold_lane) {
        const int cold_count_split = ggml_backend_sched_find_split_containing(sched, region.cold_lane.count);
        if (cold_count_split >= first_lane_split) {
            *reason = "count tensors are not in the serial prefix";
            return false;
        }
    }

    std::vector<const ggml_backend_sched_moe_hot_cache_parallel_lane_info *> ordered_lanes;
    ordered_lanes.reserve(region.hot_lanes.size() + (region.has_cold_lane ? 1 : 0));
    for (const auto & lane : region.hot_lanes) {
        ordered_lanes.push_back(&lane);
    }
    if (region.has_cold_lane) {
        ordered_lanes.push_back(&region.cold_lane);
    }
    std::sort(ordered_lanes.begin(), ordered_lanes.end(), [](const auto * a, const auto * b) {
        return a->split_begin < b->split_begin;
    });

    if (ordered_lanes.empty()) {
        *reason = "count tensors are not in the serial prefix";
        return false;
    }

    for (size_t i = 0; i < ordered_lanes.size(); ++i) {
        const auto & lane = *ordered_lanes[i];
        if (lane.split_begin > lane.split_end || lane.split_end >= region.join_split) {
            *reason = "region split order does not contain two contiguous lanes before join";
            return false;
        }
        if (i > 0 && ordered_lanes[i - 1]->split_end + 1 != lane.split_begin) {
            *reason = "region split order does not contain two contiguous lanes before join";
            return false;
        }
    }

    const bool allow_join_bridge =
        (region.flags & GGML_BACKEND_SCHED_MOE_HOT_CACHE_PARALLEL_FLAG_ALLOW_JOIN_BRIDGE) != 0;
    const int join_gap = last_lane_split >= 0 ? region.join_split - last_lane_split : -1;
    const bool join_is_contiguous = join_gap == 1;
    const bool join_has_single_bridge = allow_join_bridge && join_gap == 2;

    if (!join_is_contiguous && !join_has_single_bridge) {
        *reason = "region split order does not contain two contiguous lanes before join";
        return false;
    }

    std::vector<int> lane_backend_ids;
    lane_backend_ids.reserve(ordered_lanes.size());

    for (const auto * lane : ordered_lanes) {
        const int backend_id = sched->splits[lane->split_begin].backend_id;
        for (int seen_backend_id : lane_backend_ids) {
            if (backend_id == seen_backend_id) {
                *reason = "hot and cold lanes resolved to the same backend";
                return false;
            }
        }
        lane_backend_ids.push_back(backend_id);

        for (int split_id = lane->split_begin; split_id <= lane->split_end; ++split_id) {
            if (sched->splits[split_id].backend_id != backend_id) {
                *reason = lane->hot ? "hot lane spans multiple backends" : "cold lane spans multiple backends";
                return false;
            }
        }

        if (lane->hot && !ggml_backend_sched_backend_is_cuda(sched->backends[backend_id])) {
            *reason = "hot lane is not on a CUDA backend";
            return false;
        }
        if (!lane->hot && !ggml_backend_sched_backend_is_cpu(sched->backends[backend_id])) {
            *reason = "cold lane is not on a CPU backend";
            return false;
        }
    }

    region.hot_split_begin = region.hot_lanes.front().split_begin;
    region.hot_split_end = region.hot_lanes.front().split_end;
    region.cold_split_begin = region.has_cold_lane ? region.cold_lane.split_begin : -1;
    region.cold_split_end = region.has_cold_lane ? region.cold_lane.split_end : -1;

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
            return int64_t(2);
        }

        char * end = nullptr;
        const long long parsed = strtoll(env, &end, 10);
        if (end == env || parsed < 0) {
            return int64_t(2);
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

    std::vector<int64_t> hot_counts(region.hot_lanes.size(), 0);
    int64_t hot_count_total = 0;
    for (size_t i = 0; i < region.hot_lanes.size(); ++i) {
        hot_counts[i] = ggml_backend_sched_read_f32_count(sched, region.hot_lanes[i].count);
        if (hot_counts[i] < 0) {
            if (!ggml_backend_sched_moe_parallel_fail_or_fallback(sched, region, "count readback failed")) {
                return GGML_STATUS_FAILED;
            }
            serial_fallback = true;
            return GGML_STATUS_SUCCESS;
        }
        hot_count_total += hot_counts[i];
    }

    const int64_t cold_count = region.has_cold_lane
        ? ggml_backend_sched_read_f32_count(sched, region.cold_lane.count)
        : 0;

    if (cold_count < 0) {
        if (!ggml_backend_sched_moe_parallel_fail_or_fallback(sched, region, "count readback failed")) {
            return GGML_STATUS_FAILED;
        }
        serial_fallback = true;
        return GGML_STATUS_SUCCESS;
    }

    const int64_t active_count_total = hot_count_total + cold_count;
    if (region.mode != 2 && active_count_total > 0) {
        const int64_t min_slots = ggml_backend_sched_moe_parallel_auto_min_slots();
        if (active_count_total < min_slots) {
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
        perf.parallel_hot_count = hot_count_total > 0 ? (uint64_t) hot_count_total : 0;
        perf.parallel_cold_count = cold_count > 0 ? (uint64_t) cold_count : 0;
        ggml_backend_sched_moe_parallel_fill_split_debug(sched, region, perf);
    }

    const int64_t region_start_us = collect_perf ? ggml_time_us() : 0;

    struct runtime_lane {
        const ggml_backend_sched_moe_hot_cache_parallel_lane_info * lane;
        int64_t count;
        ggml_backend_sched_compute_state state;
        bool worker_started;
        size_t worker_index;
        int64_t start_us;
        int64_t end_us;
    };

    std::vector<runtime_lane> runtime_lanes;
    runtime_lanes.reserve(region.hot_lanes.size() + (region.has_cold_lane ? 1 : 0));

    for (size_t i = 0; i < region.hot_lanes.size(); ++i) {
        if (hot_counts[i] > 0) {
            runtime_lanes.push_back({ &region.hot_lanes[i], hot_counts[i], {}, false, 0, 0, 0 });
            if (collect_perf) {
                perf.parallel_hot_launches++;
            }
        } else if (collect_perf) {
            perf.parallel_hot_skips_zero++;
        }
    }

    if (region.has_cold_lane && cold_count > 0) {
        runtime_lanes.push_back({ &region.cold_lane, cold_count, {}, false, 0, 0, 0 });
        if (collect_perf) {
            perf.parallel_cold_launches = 1;
        }
    } else if (region.has_cold_lane && collect_perf) {
        perf.parallel_cold_skips_zero = 1;
    }

    const auto wait_started_workers = [&]() -> enum ggml_status {
        enum ggml_status result = GGML_STATUS_SUCCESS;
        for (auto & lane : runtime_lanes) {
            if (!lane.worker_started) {
                continue;
            }
            int64_t ignored_end_us = 0;
            const enum ggml_status ec = ggml_backend_sched_moe_parallel_worker_wait(
                    sched,
                    lane.worker_index,
                    ignored_end_us);
            if (ec != GGML_STATUS_SUCCESS && result == GGML_STATUS_SUCCESS) {
                result = ec;
            }
        }
        return result;
    };

    int main_lane_index = -1;
    for (size_t i = 0; i < runtime_lanes.size(); ++i) {
        if (runtime_lanes[i].lane->hot) {
            main_lane_index = (int) i;
            break;
        }
    }
    if (main_lane_index < 0 && !runtime_lanes.empty()) {
        main_lane_index = 0;
    }

    size_t worker_index = 0;
    for (size_t i = 0; i < runtime_lanes.size(); ++i) {
        if ((int) i == main_lane_index) {
            continue;
        }

        auto & lane = runtime_lanes[i];
        if (collect_perf) {
            lane.start_us = ggml_time_us();
        }

        enum ggml_status ec = ggml_backend_sched_compute_split_inputs(
                sched, &sched->splits[lane.lane->split_begin], lane.state);
        if (ec != GGML_STATUS_SUCCESS) {
            wait_started_workers();
            return ec;
        }

        lane.worker_index = worker_index++;
        ggml_backend_sched_moe_parallel_worker_start(
                sched,
                lane.worker_index,
                lane.lane->split_begin,
                lane.lane->split_end,
                lane.state);
        lane.worker_started = true;
    }

    if (main_lane_index >= 0) {
        auto & lane = runtime_lanes[(size_t) main_lane_index];
        if (collect_perf) {
            lane.start_us = ggml_time_us();
        }

        enum ggml_status ec = ggml_backend_sched_compute_split_inputs(
                sched, &sched->splits[lane.lane->split_begin], lane.state);
        if (ec != GGML_STATUS_SUCCESS) {
            wait_started_workers();
            return ec;
        }

        ec = ggml_backend_sched_compute_split_range(
                sched,
                lane.lane->split_begin,
                lane.lane->split_end,
                lane.state,
                /* use_callback = */ false,
                /* first_inputs_prepared = */ true);
        if (ec != GGML_STATUS_SUCCESS) {
            wait_started_workers();
            return ec;
        }

        ggml_backend_synchronize(sched->backends[sched->splits[lane.lane->split_begin].backend_id]);
        if (collect_perf) {
            lane.end_us = ggml_time_us();
        }
    }

    const int64_t join_wait_start_us = collect_perf ? ggml_time_us() : 0;

    enum ggml_status worker_status = wait_started_workers();
    if (worker_status != GGML_STATUS_SUCCESS) {
        return worker_status;
    }

    for (auto & lane : runtime_lanes) {
        if (!lane.worker_started) {
            continue;
        }
        ggml_backend_synchronize(sched->backends[sched->splits[lane.lane->split_begin].backend_id]);
        if (collect_perf) {
            lane.end_us = ggml_time_us();
        }
    }

    const int64_t join_wait_end_us = collect_perf ? ggml_time_us() : 0;

    for (size_t i = 0; i < region.hot_lanes.size(); ++i) {
        if (hot_counts[i] == 0 && !ggml_backend_sched_zero_tensor(region.hot_lanes[i].output)) {
            if (!ggml_backend_sched_moe_parallel_fail_or_fallback(sched, region, "failed to zero skipped hot lane")) {
                return GGML_STATUS_FAILED;
            }
            serial_fallback = true;
            return GGML_STATUS_SUCCESS;
        }
    }

    if (region.has_cold_lane && cold_count == 0 && !ggml_backend_sched_zero_tensor(region.cold_lane.output)) {
        if (!ggml_backend_sched_moe_parallel_fail_or_fallback(sched, region, "failed to zero skipped cold lane")) {
            return GGML_STATUS_FAILED;
        }
        serial_fallback = true;
        return GGML_STATUS_SUCCESS;
    }

    if (collect_perf) {
        const int64_t region_end_us = ggml_time_us();

        perf.parallel_region_wall_time_us = region_end_us > region_start_us ? (uint64_t) (region_end_us - region_start_us) : 0;

        for (const auto & lane : runtime_lanes) {
            const uint64_t wall_us =
                lane.start_us > 0 && lane.end_us > lane.start_us ? (uint64_t) (lane.end_us - lane.start_us) : 0;
            if (lane.lane->hot) {
                perf.parallel_hot_lane_wall_time_us += wall_us;
            } else {
                perf.parallel_cold_lane_wall_time_us += wall_us;
            }
        }
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

        int region_split_begin = -1;
        int region_split_end = -1;
        for (const auto & lane : region.hot_lanes) {
            region_split_begin = region_split_begin < 0 ? lane.split_begin : std::min(region_split_begin, lane.split_begin);
            region_split_end = std::max(region_split_end, lane.split_end);
        }
        if (region.has_cold_lane) {
            region_split_begin = region_split_begin < 0 ? region.cold_lane.split_begin : std::min(region_split_begin, region.cold_lane.split_begin);
            region_split_end = std::max(region_split_end, region.cold_lane.split_end);
        }

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
            split_id = region_split_end + 1;
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
    struct ggml_tensor * hot_counts[] = { hot_count };
    struct ggml_tensor * hot_starts[] = { hot_start };
    struct ggml_tensor * hot_ends[] = { hot_end };
    struct ggml_tensor * hot_outputs[] = { hot_output };

    ggml_backend_sched_moe_hot_cache_multi_parallel_region(
            sched,
            layer,
            mode,
            max_slots,
            1,
            hot_counts,
            hot_starts,
            hot_ends,
            hot_outputs,
            cold_count,
            cold_start,
            cold_end,
            cold_output,
            join,
            flags);
}

void ggml_backend_sched_moe_hot_cache_multi_parallel_region(
        ggml_backend_sched_t sched,
        int32_t layer,
        int mode,
        int64_t max_slots,
        int n_hot_lanes,
        struct ggml_tensor ** hot_counts,
        struct ggml_tensor ** hot_starts,
        struct ggml_tensor ** hot_ends,
        struct ggml_tensor ** hot_outputs,
        struct ggml_tensor * cold_count,
        struct ggml_tensor * cold_start,
        struct ggml_tensor * cold_end,
        struct ggml_tensor * cold_output,
        struct ggml_tensor * join,
        uint32_t flags) {
    GGML_ASSERT(sched);

    if (mode == 0 || n_hot_lanes <= 0) {
        return;
    }

    ggml_backend_sched_moe_hot_cache_parallel_region_info region = {};
    region.layer = layer;
    region.mode = mode;
    region.max_slots = max_slots;
    region.hot_count = n_hot_lanes > 0 ? hot_counts[0] : nullptr;
    region.cold_count = cold_count;
    region.hot_start = n_hot_lanes > 0 ? hot_starts[0] : nullptr;
    region.hot_end = n_hot_lanes > 0 ? hot_ends[0] : nullptr;
    region.hot_output = n_hot_lanes > 0 ? hot_outputs[0] : nullptr;
    region.cold_start = cold_start;
    region.cold_end = cold_end;
    region.cold_output = cold_output;
    region.join = join;
    region.flags = flags;
    region.has_cold_lane = cold_count != nullptr && cold_start != nullptr && cold_end != nullptr && cold_output != nullptr;
    region.hot_split_begin = -1;
    region.hot_split_end = -1;
    region.cold_split_begin = -1;
    region.cold_split_end = -1;
    region.join_split = -1;

    for (int i = 0; i < n_hot_lanes; ++i) {
        if (hot_counts[i] == nullptr || hot_starts[i] == nullptr || hot_ends[i] == nullptr || hot_outputs[i] == nullptr) {
            continue;
        }

        ggml_backend_sched_moe_hot_cache_parallel_lane_info lane = {};
        lane.index = i;
        lane.hot = true;
        lane.count = hot_counts[i];
        lane.start = hot_starts[i];
        lane.end = hot_ends[i];
        lane.output = hot_outputs[i];
        lane.split_begin = -1;
        lane.split_end = -1;
        region.hot_lanes.push_back(lane);
    }

    if (region.has_cold_lane) {
        region.cold_lane.index = -1;
        region.cold_lane.hot = false;
        region.cold_lane.count = cold_count;
        region.cold_lane.start = cold_start;
        region.cold_lane.end = cold_end;
        region.cold_lane.output = cold_output;
        region.cold_lane.split_begin = -1;
        region.cold_lane.split_end = -1;
    }

    if (region.hot_lanes.empty()) {
        return;
    }

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
