#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Memory/DDRController.h"
#include "Memory/MemoryDefs.h"

using namespace Memory;

namespace {

constexpr double kFrequencyMHz = 200.0;
constexpr size_t kDefaultHeadDim = 128;
constexpr size_t kDefaultBytesPerElem = 2;
constexpr size_t kDefaultHeadGroups = 8;
constexpr size_t kDefaultHeadsPerGroup = 4;
constexpr size_t kDefaultPredictPEs = 2048;
constexpr size_t kDefaultComputePEs = 256;
constexpr size_t kDefaultQueueDepth = 8;
constexpr double kDefaultHBMBandwidthGBps = 400.0;
constexpr size_t kDefaultInt4VectorBytes = 64;
constexpr double kDefaultURAMBandwidthGBps = 1000.0;
constexpr size_t kDefaultTFUThroughput = 64;
constexpr uint32_t kDefaultTcl = 14;
constexpr uint32_t kDefaultTrcd = 14;
constexpr uint32_t kDefaultTrp = 14;
constexpr uint32_t kDefaultTburst = 4;
constexpr uint32_t kDefaultBanksPerChannel = 16;
constexpr uint32_t kDefaultRowSizeBytes = 8192;
constexpr size_t kMaxInflightRequests = 64;
constexpr size_t kDenseSampleTokens = 2048;

struct ExperimentSweepConfig {
    std::vector<size_t> context_lengths{8192, 16384, 32768, 65536, 131072};
    std::vector<double> sparsities{0.01, 0.05, 0.10};
    size_t head_dim = kDefaultHeadDim;
    size_t bytes_per_elem = kDefaultBytesPerElem;
    size_t num_head_groups = kDefaultHeadGroups;
    size_t heads_per_group = kDefaultHeadsPerGroup;
    size_t predict_pes = kDefaultPredictPEs;
    size_t compute_pes = kDefaultComputePEs;
    size_t ddr_channels = 8;
    size_t max_queue_size = kDefaultQueueDepth;
    std::string csv_path;
};

struct PipelineConfigLite {
    double frequency_mhz = kFrequencyMHz;
    size_t context_length = 0;
    size_t head_dim = kDefaultHeadDim;
    size_t bytes_per_elem = kDefaultBytesPerElem;
    double sparsity_ratio = 0.0;
    size_t num_head_groups = kDefaultHeadGroups;
    size_t heads_per_group = kDefaultHeadsPerGroup;
    double hbm_scan_bandwidth_gbps = kDefaultHBMBandwidthGBps;
    size_t int4_vector_size_bytes = kDefaultInt4VectorBytes;
    size_t predict_pes = kDefaultPredictPEs;
    double uram_bandwidth_gbps = kDefaultURAMBandwidthGBps;
    size_t tfu_throughput_per_cycle = kDefaultTFUThroughput;
    size_t compute_pes = kDefaultComputePEs;
    size_t max_queue_size = kDefaultQueueDepth;
};

struct DdrSimulationResult {
    size_t total_cycles = 0;
    size_t total_bytes = 0;
};

struct TaskTrace {
    size_t p_start = 0, p_end = 0;
    size_t f_start = 0, f_end = 0;
    size_t c_start = 0, c_end = 0;
};

struct PipelineRunResult {
    size_t total_cycles = 0;
    size_t total_bubbles = 0;
    std::vector<TaskTrace> traces;
};

struct MethodResult {
    std::string method;
    size_t context_length = 0;
    double sparsity = 0.0;
    size_t total_cycles = 0;
    double latency_us = 0.0;
    double tokens_per_second = 0.0;
    double speedup_vs_dense = 0.0;
    double effective_ddr_bandwidth_gbps = 0.0;
    double bubble_rate_pct = 0.0;
    double p_util_pct = 0.0;
    double f_util_pct = 0.0;
    double c_util_pct = 0.0;
    double avg_p_cycles = 0.0;
    double avg_f_cycles = 0.0;
    double avg_c_cycles = 0.0;
};

std::vector<size_t> parse_size_list(const std::string& text) {
    std::vector<size_t> values;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) values.push_back(static_cast<size_t>(std::stoull(token)));
    }
    return values;
}

std::vector<double> parse_double_list(const std::string& text) {
    std::vector<double> values;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) values.push_back(std::stod(token));
    }
    return values;
}

ExperimentSweepConfig parse_args(int argc, char** argv) {
    ExperimentSweepConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const std::string& flag) {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + flag);
            return std::string(argv[++i]);
        };

        if (arg == "--contexts") {
            config.context_lengths = parse_size_list(require_value(arg));
        } else if (arg == "--sparsities") {
            config.sparsities = parse_double_list(require_value(arg));
        } else if (arg == "--ddr-channels") {
            config.ddr_channels = static_cast<size_t>(std::stoull(require_value(arg)));
        } else if (arg == "--head-groups") {
            config.num_head_groups = static_cast<size_t>(std::stoull(require_value(arg)));
        } else if (arg == "--heads-per-group") {
            config.heads_per_group = static_cast<size_t>(std::stoull(require_value(arg)));
        } else if (arg == "--queue-depth") {
            config.max_queue_size = static_cast<size_t>(std::stoull(require_value(arg)));
        } else if (arg == "--csv") {
            config.csv_path = require_value(arg);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: experiment_end_to_end_decode [--contexts 8192,16384,32768,65536,131072]"
                      << " [--sparsities 0.01,0.05,0.10] [--ddr-channels 8]"
                      << " [--head-groups 8] [--heads-per-group 4] [--queue-depth 8]"
                      << " [--csv result.csv]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    return config;
}

PipelineConfigLite make_pipeline_config(const ExperimentSweepConfig& sweep, size_t context_length, double sparsity) {
    PipelineConfigLite cfg;
    cfg.context_length = context_length;
    cfg.head_dim = sweep.head_dim;
    cfg.bytes_per_elem = sweep.bytes_per_elem;
    cfg.sparsity_ratio = sparsity;
    cfg.num_head_groups = sweep.num_head_groups;
    cfg.heads_per_group = sweep.heads_per_group;
    cfg.predict_pes = sweep.predict_pes;
    cfg.compute_pes = sweep.compute_pes;
    cfg.max_queue_size = sweep.max_queue_size;
    return cfg;
}

DDRController::Config make_ddr_config(size_t ddr_channels, BankMappingStrategy mapping_strategy) {
    DDRController::Config cfg;
    cfg.tCL = kDefaultTcl;
    cfg.tRCD = kDefaultTrcd;
    cfg.tRP = kDefaultTrp;
    cfg.tBURST = kDefaultTburst;
    cfg.num_channels = static_cast<uint32_t>(ddr_channels);
    cfg.num_banks_per_channel = kDefaultBanksPerChannel;
    cfg.row_size_bytes = kDefaultRowSizeBytes;
    cfg.mapping_strategy = mapping_strategy;
    return cfg;
}

size_t estimate_predict_cycles(const PipelineConfigLite& cfg) {
    const size_t total_scan_bytes = cfg.context_length * cfg.heads_per_group * cfg.int4_vector_size_bytes;
    const double hbm_bytes_per_cycle = cfg.hbm_scan_bandwidth_gbps / (cfg.frequency_mhz / 1000.0);
    const size_t hbm_cycles = static_cast<size_t>(std::ceil(total_scan_bytes / hbm_bytes_per_cycle));
    const size_t compute_cycles = static_cast<size_t>(std::ceil(
        static_cast<double>(cfg.context_length * cfg.head_dim * cfg.heads_per_group) / cfg.predict_pes)) + 44;
    const size_t tfu_cycles = static_cast<size_t>(std::ceil(static_cast<double>(cfg.context_length) / cfg.tfu_throughput_per_cycle));
    const size_t num_candidates = std::max<size_t>(1, static_cast<size_t>(std::ceil(cfg.context_length * cfg.sparsity_ratio)));
    const size_t candidate_bytes = num_candidates * 8;
    const double uram_bytes_per_cycle = cfg.uram_bandwidth_gbps / (cfg.frequency_mhz / 1000.0);
    const size_t uram_cycles = static_cast<size_t>(std::ceil(candidate_bytes / uram_bytes_per_cycle));
    return std::max({hbm_cycles, compute_cycles, tfu_cycles, uram_cycles}) + 100;
}

size_t estimate_compute_cycles(const PipelineConfigLite& cfg, size_t kv_tokens) {
    const double total_macs = static_cast<double>(2ULL * kv_tokens * cfg.head_dim * cfg.heads_per_group);
    return static_cast<size_t>(std::ceil(total_macs / cfg.compute_pes)) + 50;
}

std::vector<size_t> generate_sparse_indices(size_t count, size_t context_length, std::mt19937& gen) {
    count = std::min(count, context_length);
    if (count == 0 || context_length == 0) return {};

    std::vector<uint8_t> used(context_length, 0);
    std::vector<size_t> indices;
    indices.reserve(count);

    const size_t min_cluster_len = 4;
    const size_t max_cluster_len = 32;
    const size_t base_cluster_len = std::min(
        max_cluster_len,
        std::max(min_cluster_len, static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(count)))) / 2));
    std::uniform_int_distribution<size_t> cluster_len_dist(
        min_cluster_len,
        std::max(min_cluster_len, base_cluster_len));

    while (indices.size() < count) {
        const size_t remaining = count - indices.size();
        const size_t cluster_len = std::min(remaining, cluster_len_dist(gen));
        const size_t max_start = context_length > cluster_len ? (context_length - cluster_len) : 0;
        std::uniform_int_distribution<size_t> start_dist(0, max_start);
        const size_t start = start_dist(gen);

        for (size_t offset = 0; offset < cluster_len && indices.size() < count; ++offset) {
            const size_t idx = start + offset;
            if (idx >= context_length || used[idx]) continue;
            used[idx] = 1;
            indices.push_back(idx);
        }
    }

    return indices;
}

std::vector<MemoryRequest> build_sparse_workload(
    size_t group_id,
    size_t heads_per_group,
    size_t context_length,
    size_t token_kv_size,
    size_t selected_tokens,
    size_t ddr_channels,
    bool merge_adjacent) {

    std::mt19937 gen(2026 + static_cast<uint32_t>(group_id));
    const size_t bytes_per_head = context_length * token_kv_size;
    const size_t bank_rotation_padding_bytes = 64 * ddr_channels * kDefaultBanksPerChannel;
    const size_t padded_head_stride = bytes_per_head + bank_rotation_padding_bytes;
    const auto shared_indices = generate_sparse_indices(selected_tokens, context_length, gen);
    std::vector<std::vector<size_t>> per_head_indices(heads_per_group, shared_indices);

    std::vector<size_t> addresses;
    addresses.reserve(heads_per_group * selected_tokens);
    for (size_t round = 0; round < selected_tokens; ++round) {
        for (size_t head = 0; head < heads_per_group; ++head) {
            if (round >= per_head_indices[head].size()) continue;
            const size_t head_base = (group_id * heads_per_group + head) * padded_head_stride;
            addresses.push_back(head_base + per_head_indices[head][round] * token_kv_size);
        }
    }

    std::vector<MemoryRequest> workload;
    workload.reserve(addresses.size());
    size_t request_id = 0;

    if (!merge_adjacent || addresses.empty()) {
        for (size_t address : addresses) {
            workload.emplace_back(request_id++, address, token_kv_size, RequestType::READ, 0);
        }
        return workload;
    }

    std::sort(addresses.begin(), addresses.end());

    size_t burst_start = addresses.front();
    size_t burst_size = token_kv_size;
    for (size_t i = 1; i < addresses.size(); ++i) {
        if (addresses[i] == addresses[i - 1] + token_kv_size) {
            burst_size += token_kv_size;
            continue;
        }
        workload.emplace_back(request_id++, burst_start, burst_size, RequestType::READ, 0);
        burst_start = addresses[i];
        burst_size = token_kv_size;
    }
    workload.emplace_back(request_id++, burst_start, burst_size, RequestType::READ, 0);
    return workload;
}

std::vector<MemoryRequest> build_dense_sample_workload(
    size_t group_id,
    size_t heads_per_group,
    size_t context_length,
    size_t token_kv_size,
    size_t sample_tokens) {

    const size_t bytes_per_head = context_length * token_kv_size;
    std::vector<MemoryRequest> workload;
    workload.reserve(heads_per_group * sample_tokens);
    size_t request_id = 0;

    for (size_t head = 0; head < heads_per_group; ++head) {
        const size_t head_base = (group_id * heads_per_group + head) * bytes_per_head;
        for (size_t token = 0; token < sample_tokens; ++token) {
            workload.emplace_back(request_id++, head_base + token * token_kv_size, token_kv_size, RequestType::READ, 0);
        }
    }

    return workload;
}

std::vector<MemoryRequest> interleave_workloads(const std::vector<std::vector<MemoryRequest>>& workloads) {
    size_t total_requests = 0;
    for (const auto& workload : workloads) total_requests += workload.size();

    std::vector<MemoryRequest> merged;
    merged.reserve(total_requests);
    std::vector<size_t> offsets(workloads.size(), 0);
    size_t next_request_id = 0;

    bool has_progress = true;
    while (has_progress) {
        has_progress = false;
        for (size_t workload_id = 0; workload_id < workloads.size(); ++workload_id) {
            if (offsets[workload_id] >= workloads[workload_id].size()) continue;

            const auto& request = workloads[workload_id][offsets[workload_id]++];
            merged.emplace_back(next_request_id++, request.address, request.size_bytes, request.type, 0);
            has_progress = true;
        }
    }

    return merged;
}

DdrSimulationResult run_ddr_simulation(DDRController& ddr, std::vector<MemoryRequest> workload) {
    size_t current_cycle = 0;
    size_t request_index = 0;
    size_t completed_count = 0;
    size_t inflight = 0;
    size_t total_bytes = 0;

    for (const auto& request : workload) total_bytes += request.size_bytes;

    while (completed_count < workload.size()) {
        while (request_index < workload.size() && inflight < kMaxInflightRequests) {
            workload[request_index].arrival_cycle = current_cycle;
            if (!ddr.send_request(workload[request_index])) break;
            ++request_index;
            ++inflight;
        }

        ddr.step(current_cycle);
        auto completed = ddr.pop_completed_requests();
        if (!completed.empty()) {
            completed_count += completed.size();
            inflight -= completed.size();
        }

        ++current_cycle;
        if (current_cycle > 100000000) throw std::runtime_error("DDR simulation timeout");
    }

    return {current_cycle, total_bytes};
}

size_t estimate_dense_fetch_cycles_shared(const PipelineConfigLite& cfg, size_t ddr_channels) {
    const size_t token_kv_size = cfg.head_dim * cfg.bytes_per_elem * 2;
    const size_t sample_tokens = std::min(cfg.context_length, kDenseSampleTokens);

    std::vector<std::vector<MemoryRequest>> per_group_workloads;
    per_group_workloads.reserve(cfg.num_head_groups);
    for (size_t group_id = 0; group_id < cfg.num_head_groups; ++group_id) {
        per_group_workloads.push_back(
            build_dense_sample_workload(group_id, cfg.heads_per_group, cfg.context_length, token_kv_size, sample_tokens));
    }

    auto workload = interleave_workloads(per_group_workloads);
    DDRController ddr(make_ddr_config(ddr_channels, BankMappingStrategy::LINEAR));
    auto sample_result = run_ddr_simulation(ddr, std::move(workload));
    const double scale = static_cast<double>(cfg.context_length) / sample_tokens;
    return static_cast<size_t>(std::ceil(sample_result.total_cycles * scale));
}

size_t estimate_sparse_fetch_cycles_shared(
    const PipelineConfigLite& cfg,
    size_t ddr_channels,
    BankMappingStrategy mapping,
    bool merge_adjacent) {

    const size_t token_kv_size = cfg.head_dim * cfg.bytes_per_elem * 2;
    const size_t selected_tokens = std::max<size_t>(1, static_cast<size_t>(std::ceil(cfg.context_length * cfg.sparsity_ratio)));

    std::vector<std::vector<MemoryRequest>> per_group_workloads;
    per_group_workloads.reserve(cfg.num_head_groups);
    for (size_t group_id = 0; group_id < cfg.num_head_groups; ++group_id) {
        per_group_workloads.push_back(build_sparse_workload(
            group_id, cfg.heads_per_group, cfg.context_length, token_kv_size, selected_tokens, ddr_channels, merge_adjacent));
    }

    auto workload = interleave_workloads(per_group_workloads);
    DDRController ddr(make_ddr_config(ddr_channels, mapping));
    return run_ddr_simulation(ddr, std::move(workload)).total_cycles;
}

size_t estimate_sparse_fetch_cycles_per_group(
    size_t group_id,
    const PipelineConfigLite& cfg,
    size_t ddr_channels,
    BankMappingStrategy mapping,
    bool merge_adjacent) {

    const size_t token_kv_size = cfg.head_dim * cfg.bytes_per_elem * 2;
    const size_t selected_tokens = std::max<size_t>(1, static_cast<size_t>(std::ceil(cfg.context_length * cfg.sparsity_ratio)));
    auto workload = build_sparse_workload(
        group_id, cfg.heads_per_group, cfg.context_length, token_kv_size, selected_tokens, ddr_channels, merge_adjacent);
    DDRController ddr(make_ddr_config(ddr_channels, mapping));
    return run_ddr_simulation(ddr, std::move(workload)).total_cycles;
}

double cycles_to_microseconds(size_t cycles) {
    return static_cast<double>(cycles) / kFrequencyMHz;
}

double cycles_to_tokens_per_second(size_t cycles) {
    const double seconds = static_cast<double>(cycles) / (kFrequencyMHz * 1e6);
    return seconds > 0.0 ? 1.0 / seconds : 0.0;
}

double bytes_to_effective_bandwidth(size_t total_bytes, size_t cycles) {
    if (cycles == 0) return 0.0;
    const double time_ns = static_cast<double>(cycles) * (1000.0 / kFrequencyMHz);
    return static_cast<double>(total_bytes) / time_ns;
}

MethodResult finalize_result(const std::string& method, size_t context_length, double sparsity, size_t total_cycles,
                             size_t total_ddr_bytes, double bubble_rate_pct,
                             double p_util_pct, double f_util_pct, double c_util_pct,
                             double avg_p_cycles, double avg_f_cycles, double avg_c_cycles) {
    MethodResult result;
    result.method = method;
    result.context_length = context_length;
    result.sparsity = sparsity;
    result.total_cycles = total_cycles;
    result.latency_us = cycles_to_microseconds(total_cycles);
    result.tokens_per_second = cycles_to_tokens_per_second(total_cycles);
    result.effective_ddr_bandwidth_gbps = bytes_to_effective_bandwidth(total_ddr_bytes, total_cycles);
    result.bubble_rate_pct = bubble_rate_pct;
    result.p_util_pct = p_util_pct;
    result.f_util_pct = f_util_pct;
    result.c_util_pct = c_util_pct;
    result.avg_p_cycles = avg_p_cycles;
    result.avg_f_cycles = avg_f_cycles;
    result.avg_c_cycles = avg_c_cycles;
    return result;
}

MethodResult run_dense_baseline(const ExperimentSweepConfig& sweep, size_t context_length, double sparsity) {
    const auto cfg = make_pipeline_config(sweep, context_length, sparsity);
    const size_t compute_cycles = estimate_compute_cycles(cfg, context_length);
    const size_t token_kv_size = cfg.head_dim * cfg.bytes_per_elem * 2;
    const size_t total_ddr_bytes = cfg.num_head_groups * cfg.heads_per_group * context_length * token_kv_size;

    const size_t total_fetch_cycles = estimate_dense_fetch_cycles_shared(cfg, sweep.ddr_channels);

    const size_t total_compute_cycles = cfg.num_head_groups * compute_cycles;
    const size_t total_cycles = total_fetch_cycles + total_compute_cycles;

    return finalize_result(
        "Dense-FullKV",
        context_length,
        sparsity,
        total_cycles,
        total_ddr_bytes,
        0.0,
        0.0,
        100.0 * total_fetch_cycles / total_cycles,
        100.0 * total_compute_cycles / total_cycles,
        0.0,
        static_cast<double>(total_fetch_cycles) / cfg.num_head_groups,
        static_cast<double>(compute_cycles));
}

MethodResult run_sparse_serial(const ExperimentSweepConfig& sweep, size_t context_length, double sparsity,
                               BankMappingStrategy mapping, bool merge_adjacent, const std::string& label) {
    const auto cfg = make_pipeline_config(sweep, context_length, sparsity);
    const size_t predict_cycles = estimate_predict_cycles(cfg);
    const size_t selected_tokens = std::max<size_t>(1, static_cast<size_t>(std::ceil(context_length * sparsity)));
    const size_t compute_cycles = estimate_compute_cycles(cfg, selected_tokens);
    const size_t token_kv_size = cfg.head_dim * cfg.bytes_per_elem * 2;
    const size_t total_ddr_bytes = cfg.num_head_groups * cfg.heads_per_group * selected_tokens * token_kv_size;

    const size_t total_fetch_cycles = estimate_sparse_fetch_cycles_shared(
        cfg, sweep.ddr_channels, mapping, merge_adjacent);

    const size_t total_predict_cycles = cfg.num_head_groups * predict_cycles;
    const size_t total_compute_cycles = cfg.num_head_groups * compute_cycles;
    const size_t total_cycles = total_predict_cycles + total_fetch_cycles + total_compute_cycles;

    return finalize_result(
        label,
        context_length,
        sparsity,
        total_cycles,
        total_ddr_bytes,
        0.0,
        100.0 * total_predict_cycles / total_cycles,
        100.0 * total_fetch_cycles / total_cycles,
        100.0 * total_compute_cycles / total_cycles,
        static_cast<double>(predict_cycles),
        static_cast<double>(total_fetch_cycles) / cfg.num_head_groups,
        static_cast<double>(compute_cycles));
}

PipelineRunResult simulate_pipeline(
    const std::vector<size_t>& p_latencies,
    const std::vector<size_t>& f_latencies,
    const std::vector<size_t>& c_latencies,
    size_t max_queue_size) {

    const size_t num_tasks = p_latencies.size();
    PipelineRunResult result;
    result.traces.resize(num_tasks);

    std::vector<int> pending_start;
    pending_start.reserve(num_tasks);
    for (size_t i = 0; i < num_tasks; ++i) pending_start.push_back(static_cast<int>(i));
    size_t start_index = 0;

    std::vector<int> pending_fetch;
    std::vector<int> pending_compute;

    int busy_p = -1;
    int busy_f = -1;
    int busy_c = -1;
    size_t p_free = 0;
    size_t f_free = 0;
    size_t c_free = 0;
    size_t finished = 0;

    while (finished < num_tasks) {
        ++result.total_cycles;
        bool p_active = false;
        bool f_active = false;
        bool c_active = false;

        if (busy_c != -1) {
            c_active = true;
            if (result.total_cycles >= c_free) {
                result.traces[busy_c].c_end = result.total_cycles;
                busy_c = -1;
                ++finished;
            }
        }
        if (busy_c == -1 && !pending_compute.empty()) {
            busy_c = pending_compute.front();
            pending_compute.erase(pending_compute.begin());
            c_active = true;
            result.traces[busy_c].c_start = result.total_cycles;
            c_free = result.total_cycles + c_latencies[busy_c];
        }

        if (busy_f != -1) {
            f_active = true;
            if (result.total_cycles >= f_free) {
                if (pending_compute.size() < max_queue_size) {
                    result.traces[busy_f].f_end = result.total_cycles;
                    pending_compute.push_back(busy_f);
                    busy_f = -1;
                }
            }
        }
        if (busy_f == -1 && !pending_fetch.empty()) {
            busy_f = pending_fetch.front();
            pending_fetch.erase(pending_fetch.begin());
            f_active = true;
            result.traces[busy_f].f_start = result.total_cycles;
            f_free = result.total_cycles + f_latencies[busy_f];
        }

        if (busy_p != -1) {
            p_active = true;
            if (result.total_cycles >= p_free) {
                if (pending_fetch.size() < max_queue_size) {
                    result.traces[busy_p].p_end = result.total_cycles;
                    pending_fetch.push_back(busy_p);
                    busy_p = -1;
                }
            }
        }
        if (busy_p == -1 && start_index < pending_start.size()) {
            busy_p = pending_start[start_index++];
            p_active = true;
            result.traces[busy_p].p_start = result.total_cycles;
            p_free = result.total_cycles + p_latencies[busy_p];
        }

        if (!p_active && !f_active && !c_active) ++result.total_bubbles;

        if (result.total_cycles > 100000000) throw std::runtime_error("Pipeline simulation timeout");
    }

    return result;
}

MethodResult run_sparse_pipeline(const ExperimentSweepConfig& sweep, size_t context_length, double sparsity) {
    const auto cfg = make_pipeline_config(sweep, context_length, sparsity);
    const size_t predict_cycles = estimate_predict_cycles(cfg);
    const size_t selected_tokens = std::max<size_t>(1, static_cast<size_t>(std::ceil(context_length * sparsity)));
    const size_t compute_cycles = estimate_compute_cycles(cfg, selected_tokens);
    const size_t token_kv_size = cfg.head_dim * cfg.bytes_per_elem * 2;
    const size_t total_ddr_bytes = cfg.num_head_groups * cfg.heads_per_group * selected_tokens * token_kv_size;

    std::vector<size_t> p_latencies(cfg.num_head_groups, predict_cycles);
    std::vector<size_t> c_latencies(cfg.num_head_groups, compute_cycles);
    std::vector<size_t> f_latencies;
    f_latencies.reserve(cfg.num_head_groups);
    for (size_t group = 0; group < cfg.num_head_groups; ++group) {
        f_latencies.push_back(estimate_sparse_fetch_cycles_per_group(group, cfg, sweep.ddr_channels, BankMappingStrategy::HASH_XOR, true));
    }

    auto pipeline = simulate_pipeline(p_latencies, f_latencies, c_latencies, cfg.max_queue_size);

    size_t total_p = 0;
    size_t total_f = 0;
    size_t total_c = 0;
    for (const auto& trace : pipeline.traces) {
        total_p += trace.p_end - trace.p_start;
        total_f += trace.f_end - trace.f_start;
        total_c += trace.c_end - trace.c_start;
    }

    return finalize_result(
        "Sparse-HashMergePipeline",
        context_length,
        sparsity,
        pipeline.total_cycles,
        total_ddr_bytes,
        100.0 * pipeline.total_bubbles / pipeline.total_cycles,
        100.0 * total_p / pipeline.total_cycles,
        100.0 * total_f / pipeline.total_cycles,
        100.0 * total_c / pipeline.total_cycles,
        static_cast<double>(total_p) / cfg.num_head_groups,
        static_cast<double>(total_f) / cfg.num_head_groups,
        static_cast<double>(total_c) / cfg.num_head_groups);
}

void write_csv(const std::string& path, const std::vector<MethodResult>& results) {
    std::ofstream out(path);
    out << "context_length,sparsity,method,total_cycles,latency_us,tokens_per_second,speedup_vs_dense,"
           "effective_ddr_bandwidth_gbps,bubble_rate_pct,p_util_pct,f_util_pct,c_util_pct,"
           "avg_p_cycles,avg_f_cycles,avg_c_cycles\n";
    for (const auto& result : results) {
        out << result.context_length << ','
            << result.sparsity << ','
            << result.method << ','
            << result.total_cycles << ','
            << result.latency_us << ','
            << result.tokens_per_second << ','
            << result.speedup_vs_dense << ','
            << result.effective_ddr_bandwidth_gbps << ','
            << result.bubble_rate_pct << ','
            << result.p_util_pct << ','
            << result.f_util_pct << ','
            << result.c_util_pct << ','
            << result.avg_p_cycles << ','
            << result.avg_f_cycles << ','
            << result.avg_c_cycles << '\n';
    }
}

void print_header() {
    std::cout << std::left
              << std::setw(28) << "Method"
              << std::setw(12) << "Cycles"
              << std::setw(14) << "Latency(us)"
              << std::setw(14) << "Tok/s"
              << std::setw(12) << "Speedup"
              << std::setw(14) << "DDR GB/s"
              << std::setw(12) << "Bubble%"
              << std::setw(10) << "P util"
              << std::setw(10) << "F util"
              << std::setw(10) << "C util"
              << '\n';
}

void print_result(const MethodResult& result) {
    std::cout << std::left
              << std::setw(28) << result.method
              << std::setw(12) << result.total_cycles
              << std::setw(14) << std::fixed << std::setprecision(2) << result.latency_us
              << std::setw(14) << std::fixed << std::setprecision(2) << result.tokens_per_second
              << std::setw(12) << std::fixed << std::setprecision(2) << result.speedup_vs_dense
              << std::setw(14) << std::fixed << std::setprecision(2) << result.effective_ddr_bandwidth_gbps
              << std::setw(12) << std::fixed << std::setprecision(2) << result.bubble_rate_pct
              << std::setw(10) << std::fixed << std::setprecision(1) << result.p_util_pct
              << std::setw(10) << std::fixed << std::setprecision(1) << result.f_util_pct
              << std::setw(10) << std::fixed << std::setprecision(1) << result.c_util_pct
              << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto sweep = parse_args(argc, argv);
        std::vector<MethodResult> all_results;

        std::cout << "=========================================================================\n";
        std::cout << " End-to-End Decode Attention Experiment (Hardware-Side)\n";
        std::cout << "=========================================================================\n";
        std::cout << "Methods: Dense-FullKV | Sparse-LinearNaive | Sparse-HashNoMerge | "
                     "Sparse-HashMerge | Sparse-HashMergePipeline\n";
        std::cout << "DDR Channels: " << sweep.ddr_channels
                  << " | Head Groups: " << sweep.num_head_groups
                  << " | Heads/Group: " << sweep.heads_per_group
                  << " | Queue Depth: " << sweep.max_queue_size << "\n\n";

        for (size_t context_length : sweep.context_lengths) {
            for (double sparsity : sweep.sparsities) {
                std::cout << "-----------------------------------------------------------------\n";
                std::cout << "Context Length = " << context_length
                          << ", Sparsity = " << std::fixed << std::setprecision(2) << (sparsity * 100.0)
                          << "%\n";
                std::cout << "-----------------------------------------------------------------\n";
                print_header();

                std::vector<MethodResult> case_results;
                case_results.push_back(run_dense_baseline(sweep, context_length, sparsity));
                case_results.push_back(run_sparse_serial(
                    sweep, context_length, sparsity, BankMappingStrategy::LINEAR, false, "Sparse-LinearNaive"));
                case_results.push_back(run_sparse_serial(
                    sweep, context_length, sparsity, BankMappingStrategy::HASH_XOR, false, "Sparse-HashNoMerge"));
                case_results.push_back(run_sparse_serial(
                    sweep, context_length, sparsity, BankMappingStrategy::HASH_XOR, true, "Sparse-HashMerge"));
                case_results.push_back(run_sparse_pipeline(sweep, context_length, sparsity));

                const double dense_cycles = static_cast<double>(case_results.front().total_cycles);
                for (auto& result : case_results) {
                    result.speedup_vs_dense = dense_cycles / std::max(1.0, static_cast<double>(result.total_cycles));
                    print_result(result);
                    all_results.push_back(result);
                }
                std::cout << '\n';
            }
        }

        if (!sweep.csv_path.empty()) {
            write_csv(sweep.csv_path, all_results);
            std::cout << "CSV written to: " << sweep.csv_path << '\n';
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
