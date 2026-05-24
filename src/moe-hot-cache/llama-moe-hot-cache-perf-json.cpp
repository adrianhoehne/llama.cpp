#include "llama-moe-hot-cache-perf-json.h"

#include <algorithm>
#include <sstream>

namespace {

bool non_empty_layer(const llama_moe_layer_perf_json_layer_snapshot & layer) {
    return layer.calls != 0 ||
           layer.expert_hits_total != 0 ||
           layer.total_moe_time_us != 0 ||
           layer.parallel_fallbacks != 0;
}

double per_call(uint64_t value, uint64_t calls) {
    return calls > 0 ? (double) value / (double) calls : 0.0;
}

double ratio(uint64_t value, uint64_t total) {
    return total > 0 ? (double) value / (double) total : 0.0;
}

bool has_expert_counts(const std::vector<uint64_t> & counts) {
    return std::any_of(counts.begin(), counts.end(), [](uint64_t count) {
        return count != 0;
    });
}

bool raw_counts_are_cold(const llama_moe_layer_perf_json_layer_snapshot & layer) {
    return layer.expert_hits_total != 0 &&
           !has_expert_counts(layer.hot_experts) &&
           !has_expert_counts(layer.cold_experts);
}

void write_expert_counts(std::ostringstream & out, const std::vector<uint64_t> & counts) {
    out << "[";

    bool first_expert = true;

    for (size_t ex = 0; ex < counts.size(); ++ex) {
        if (counts[ex] == 0) {
            continue;
        }

        if (!first_expert) {
            out << ",";
        }

        first_expert = false;

        out << "[" << ex << "," << counts[ex] << "]";
    }

    out << "]";
}

void write_fallback_reason(std::ostringstream & out, bool & first, const char * name, uint64_t value) {
    if (value == 0) {
        return;
    }

    if (!first) {
        out << ",";
    }

    first = false;
    out << "\"" << name << "\":" << value;
}

} // namespace

std::string llama_moe_layer_perf_json_serializer::serialize_disabled() {
    return "{\"enabled\":false,\"mode\":\"off\",\"schema\":\"llama.cpp.moe_layer_opt_perf.v1\",\"layers\":[]}";
}

std::string llama_moe_layer_perf_json_serializer::serialize(const llama_moe_layer_perf_json_snapshot & snapshot) {
    if (!snapshot.enabled || snapshot.mode == LLAMA_MOE_LAYER_PERF_MODE_OFF) {
        return serialize_disabled();
    }

    const bool full_mode = snapshot.mode == LLAMA_MOE_LAYER_PERF_MODE_FULL;

    uint64_t summary_calls = 0;
    uint64_t summary_hot_slots = 0;
    uint64_t summary_cold_slots = 0;
    uint64_t summary_total_moe_time_us = 0;
    uint64_t summary_routing_time_us = 0;
    uint64_t summary_worklist_time_us = 0;
    uint64_t summary_merge_time_us = 0;
    uint64_t summary_hot_branch_time_us = 0;
    uint64_t summary_cold_branch_time_us = 0;
    uint64_t summary_hot_expert_matmul_time_us = 0;
    uint64_t summary_cold_expert_matmul_time_us = 0;
    uint64_t summary_hot_gather_scatter_time_us = 0;
    uint64_t summary_cold_gather_scatter_time_us = 0;
    uint64_t summary_parallel_region_wall_time_us = 0;
    uint64_t summary_parallel_hot_lane_wall_time_us = 0;
    uint64_t summary_parallel_cold_lane_wall_time_us = 0;
    uint64_t summary_parallel_join_wait_time_us = 0;
    uint64_t summary_parallel_overlap_estimate_us = 0;
    uint64_t summary_parallel_hot_launches = 0;
    uint64_t summary_parallel_cold_launches = 0;
    uint64_t summary_parallel_fallbacks = 0;

    for (const auto & layer : snapshot.layers) {
        if (!non_empty_layer(layer)) {
            continue;
        }

        const bool raw_as_cold = raw_counts_are_cold(layer);

        summary_calls += layer.calls;
        summary_hot_slots += raw_as_cold ? 0 : layer.hot_slots_total;
        summary_cold_slots += raw_as_cold ? layer.expert_hits_total : layer.cold_slots_total;
        summary_total_moe_time_us += layer.total_moe_time_us;
        summary_routing_time_us += layer.routing_time_us;
        summary_worklist_time_us += layer.worklist_time_us;
        summary_merge_time_us += layer.merge_time_us;
        summary_hot_branch_time_us += layer.hot_branch_time_us;
        summary_cold_branch_time_us += layer.cold_branch_time_us;
        summary_hot_expert_matmul_time_us += layer.hot_expert_matmul_time_us;
        summary_cold_expert_matmul_time_us += layer.cold_expert_matmul_time_us;
        summary_hot_gather_scatter_time_us += layer.hot_gather_scatter_time_us;
        summary_cold_gather_scatter_time_us += layer.cold_gather_scatter_time_us;
        summary_parallel_region_wall_time_us += layer.parallel_region_wall_time_us;
        summary_parallel_hot_lane_wall_time_us += layer.parallel_hot_lane_wall_time_us;
        summary_parallel_cold_lane_wall_time_us += layer.parallel_cold_lane_wall_time_us;
        summary_parallel_join_wait_time_us += layer.parallel_join_wait_time_us;
        summary_parallel_overlap_estimate_us += layer.parallel_overlap_estimate_us;
        summary_parallel_hot_launches += layer.parallel_hot_launches;
        summary_parallel_cold_launches += layer.parallel_cold_launches;
        summary_parallel_fallbacks += layer.parallel_fallbacks;
    }

    const uint64_t summary_slots = summary_hot_slots + summary_cold_slots;

    std::ostringstream out;
    out << "{";
    out << "\"enabled\":true,";
    out << "\"mode\":\"" << llama_moe_layer_perf_mode_name(snapshot.mode) << "\",";
    out << "\"schema\":\"llama.cpp.moe_layer_opt_perf.v1\",";
    out << "\"n_expert\":" << snapshot.n_expert << ",";
    out << "\"n_expert_used\":" << snapshot.n_expert_used << ",";
    out << "\"updates\":" << snapshot.updates << ",";
    out << "\"overflow_resets\":" << snapshot.overflow_resets << ",";
    out << "\"summary\":{";
    out << "\"layer_calls\":" << summary_calls;
    out << ",\"hot_slot_ratio\":" << ratio(summary_hot_slots, summary_slots);
    out << ",\"parallel_hot_lane_wall_time_per_call_us\":" << per_call(summary_parallel_hot_lane_wall_time_us, summary_calls);
    out << ",\"parallel_cold_lane_wall_time_per_call_us\":" << per_call(summary_parallel_cold_lane_wall_time_us, summary_calls);
    out << ",\"parallel_join_wait_time_per_call_us\":" << per_call(summary_parallel_join_wait_time_us, summary_calls);
    if (full_mode) {
        out << ",\"total_moe_time_per_call_us\":" << per_call(summary_total_moe_time_us, summary_calls);
        out << ",\"routing_time_per_call_us\":" << per_call(summary_routing_time_us, summary_calls);
        out << ",\"worklist_time_per_call_us\":" << per_call(summary_worklist_time_us, summary_calls);
        out << ",\"merge_time_per_call_us\":" << per_call(summary_merge_time_us, summary_calls);
        out << ",\"hot_branch_time_per_call_us\":" << per_call(summary_hot_branch_time_us, summary_calls);
        out << ",\"cold_branch_time_per_call_us\":" << per_call(summary_cold_branch_time_us, summary_calls);
        out << ",\"hot_expert_matmul_time_per_call_us\":" << per_call(summary_hot_expert_matmul_time_us, summary_calls);
        out << ",\"cold_expert_matmul_time_per_call_us\":" << per_call(summary_cold_expert_matmul_time_us, summary_calls);
        out << ",\"hot_gather_scatter_time_per_call_us\":" << per_call(summary_hot_gather_scatter_time_us, summary_calls);
        out << ",\"cold_gather_scatter_time_per_call_us\":" << per_call(summary_cold_gather_scatter_time_us, summary_calls);
        out << ",\"parallel_region_wall_time_per_call_us\":" << per_call(summary_parallel_region_wall_time_us, summary_calls);
        out << ",\"parallel_overlap_estimate_per_call_us\":" << per_call(summary_parallel_overlap_estimate_us, summary_calls);
        out << ",\"parallel_hot_launches\":" << summary_parallel_hot_launches;
        out << ",\"parallel_cold_launches\":" << summary_parallel_cold_launches;
        out << ",\"parallel_fallbacks\":" << summary_parallel_fallbacks;
    }
    out << "},";
    out << "\"layers\":[";

    bool first_layer = true;

    for (size_t il = 0; il < snapshot.layers.size(); ++il) {
        const auto & layer = snapshot.layers[il];

        if (!non_empty_layer(layer)) {
            continue;
        }

        if (!first_layer) {
            out << ",";
        }

        first_layer = false;

        const bool raw_as_cold = raw_counts_are_cold(layer);
        const uint64_t hot_slots_total = raw_as_cold ? 0 : layer.hot_slots_total;
        const uint64_t cold_slots_total = raw_as_cold ? layer.expert_hits_total : layer.cold_slots_total;

        const double hot_slots_per_call =
            raw_as_cold ? 0.0 :
            layer.hot_worklist_calls > 0 ? (double) hot_slots_total / (double) layer.hot_worklist_calls : per_call(hot_slots_total, layer.calls);

        const double cold_slots_per_call =
            raw_as_cold ? per_call(cold_slots_total, layer.calls) :
            layer.cold_worklist_calls > 0 ? (double) cold_slots_total / (double) layer.cold_worklist_calls : per_call(cold_slots_total, layer.calls);

        const uint64_t slots_total = hot_slots_total + cold_slots_total;

        out << "{";
        out << "\"layer\":" << il << ",";
        out << "\"calls\":" << layer.calls << ",";
        out << "\"hot_slots_total\":" << hot_slots_total << ",";
        out << "\"cold_slots_total\":" << cold_slots_total << ",";
        out << "\"hot_slots_per_call\":" << hot_slots_per_call << ",";
        out << "\"cold_slots_per_call\":" << cold_slots_per_call << ",";
        out << "\"hot_slot_ratio\":" << ratio(hot_slots_total, slots_total);
        out << ",\"parallel_hot_lane_wall_time_per_call_us\":" << per_call(layer.parallel_hot_lane_wall_time_us, layer.calls);
        out << ",\"parallel_cold_lane_wall_time_per_call_us\":" << per_call(layer.parallel_cold_lane_wall_time_us, layer.calls);
        out << ",\"parallel_join_wait_time_per_call_us\":" << per_call(layer.parallel_join_wait_time_us, layer.calls);

        if (full_mode) {
            out << ",\"total_moe_time_per_call_us\":" << per_call(layer.total_moe_time_us, layer.calls);
            out << ",\"routing_time_per_call_us\":" << per_call(layer.routing_time_us, layer.calls);
            out << ",\"worklist_time_per_call_us\":" << per_call(layer.worklist_time_us, layer.calls);
            out << ",\"merge_time_per_call_us\":" << per_call(layer.merge_time_us, layer.calls);
            out << ",\"hot_branch_time_per_call_us\":" << per_call(layer.hot_branch_time_us, layer.calls);
            out << ",\"cold_branch_time_per_call_us\":" << per_call(layer.cold_branch_time_us, layer.calls);
            out << ",\"hot_expert_matmul_time_per_call_us\":" << per_call(layer.hot_expert_matmul_time_us, layer.calls);
            out << ",\"cold_expert_matmul_time_per_call_us\":" << per_call(layer.cold_expert_matmul_time_us, layer.calls);
            out << ",\"hot_gather_scatter_time_per_call_us\":" << per_call(layer.hot_gather_scatter_time_us, layer.calls);
            out << ",\"cold_gather_scatter_time_per_call_us\":" << per_call(layer.cold_gather_scatter_time_us, layer.calls);
            out << ",\"parallel_region_wall_time_per_call_us\":" << per_call(layer.parallel_region_wall_time_us, layer.calls);
            out << ",\"parallel_overlap_estimate_per_call_us\":" << per_call(layer.parallel_overlap_estimate_us, layer.calls);
            out << ",\"parallel_hot_launches\":" << layer.parallel_hot_launches;
            out << ",\"parallel_cold_launches\":" << layer.parallel_cold_launches;
            out << ",\"parallel_hot_skips_zero\":" << layer.parallel_hot_skips_zero;
            out << ",\"parallel_cold_skips_zero\":" << layer.parallel_cold_skips_zero;
            out << ",\"parallel_fallbacks\":" << layer.parallel_fallbacks;

            if (layer.parallel_fallbacks > 0) {
                bool first_reason = true;
                out << ",\"parallel_fallback_reasons\":{";
                write_fallback_reason(out, first_reason, "incomplete", layer.parallel_fallback_incomplete);
                write_fallback_reason(out, first_reason, "count_not_prefix", layer.parallel_fallback_count_not_prefix);
                write_fallback_reason(out, first_reason, "bad_split_order", layer.parallel_fallback_bad_split_order);
                write_fallback_reason(out, first_reason, "same_backend", layer.parallel_fallback_same_backend);
                write_fallback_reason(out, first_reason, "hot_spans_backends", layer.parallel_fallback_hot_spans_backends);
                write_fallback_reason(out, first_reason, "cold_spans_backends", layer.parallel_fallback_cold_spans_backends);
                write_fallback_reason(out, first_reason, "hot_not_cuda", layer.parallel_fallback_hot_not_cuda);
                write_fallback_reason(out, first_reason, "cold_not_cpu", layer.parallel_fallback_cold_not_cpu);
                write_fallback_reason(out, first_reason, "count_readback", layer.parallel_fallback_count_readback);
                write_fallback_reason(out, first_reason, "threshold", layer.parallel_fallback_threshold);
                write_fallback_reason(out, first_reason, "zero_output", layer.parallel_fallback_zero_output);
                write_fallback_reason(out, first_reason, "other", layer.parallel_fallback_other);
                out << "}";
            }

            if (layer.parallel_split_debug_samples > 0) {
                out << ",\"parallel_split_debug\":{";
                out << "\"samples\":" << layer.parallel_split_debug_samples << ",";
                out << "\"last\":{";
                out << "\"hot\":[" << layer.parallel_split_debug_hot_begin << "," << layer.parallel_split_debug_hot_end << "],";
                out << "\"cold\":[" << layer.parallel_split_debug_cold_begin << "," << layer.parallel_split_debug_cold_end << "],";
                out << "\"join\":" << layer.parallel_split_debug_join << ",";
                out << "\"counts\":{\"hot\":" << layer.parallel_split_debug_hot_count << ",\"cold\":" << layer.parallel_split_debug_cold_count << "},";
                out << "\"backends\":{\"hot\":" << layer.parallel_split_debug_hot_backend << ",\"cold\":" << layer.parallel_split_debug_cold_backend << ",\"join\":" << layer.parallel_split_debug_join_backend << "}";
                out << "}}";
            }
        }

        out << ",\"experts\":";
        write_expert_counts(out, layer.experts);
        out << ",\"hot_experts\":";
        write_expert_counts(out, layer.hot_experts);
        out << ",\"cold_experts\":";
        write_expert_counts(out, raw_as_cold ? layer.experts : layer.cold_experts);

        out << "}";
    }

    out << "]}";

    return out.str();
}
