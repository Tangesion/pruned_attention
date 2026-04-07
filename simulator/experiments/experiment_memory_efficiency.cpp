#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include "Memory/DDRController.h"
#include "Memory/MemoryDefs.h"

using namespace Memory;

// =================================================================================
// Configuration Constants
// =================================================================================
const double FIXED_SELECTION_RATIO = 0.10; // Keep 10% tokens (equiv. 90% sparsity)
const size_t HIDDEN_SIZE = 128;         // 128 channels (Head Dimension)
const size_t BYTES_PER_ELEMENT = 2;     // BF16
const size_t TOKEN_SIZE_BYTES = HIDDEN_SIZE * BYTES_PER_ELEMENT; // 256 Bytes per token
const size_t NUM_HEAD_STREAMS = 32;     // Multi-head sparse gather streams
const size_t HEAD_STRIDE_PADDING_BYTES = 0; // Naive contiguous head layout
const size_t SGU_MERGE_BYTES = 4096;           // Conservative SGU merge window
const size_t HASH_SGU_MERGE_BYTES = 8192;      // Larger window enabled by hash-aware scheduling
const size_t HASH_BUCKET_CHUNK = 4;            // Short burst per channel-bank bucket
const size_t LARGE_HASH_BUCKET_CHUNK = 4096;   // Drain large bank buckets to maximize row locality
const size_t LARGE_CONTEXT_BANK_ONLY_THRESHOLD = 256 * 1024;
const size_t DEFAULT_MAX_IN_FLIGHT_BURSTS = 256;
const size_t HASH_SCHED_MAX_IN_FLIGHT_BURSTS = 512;

// DDR4-3200ish Configuration
const DDRController::Config DDR_CONFIG = {
    .tCL = 14,          // CAS Latency
    .tRCD = 14,         // Row to Column Delay
    .tRP = 14,          // Row Precharge
    .tBURST = 4,        // Burst time for cache line (64B)
    .num_channels = 8,  // 8 Channels (High-end FPGA/Server)
    .num_banks_per_channel = 16,
    .row_size_bytes = 8 * 1024, // 8KB Row Buffer
    .mapping_strategy = BankMappingStrategy::LINEAR // Will be changed dynamically
};

// =================================================================================
// Helper: Workload Generation
// =================================================================================
std::vector<size_t> generate_indices(double selection_ratio, size_t total_tokens) {
    size_t num_selected = static_cast<size_t>(total_tokens * selection_ratio);
    num_selected = std::min(num_selected, total_tokens);

    std::vector<size_t> indices;
    indices.reserve(num_selected);
    if (num_selected == 0 || total_tokens == 0) {
        return indices;
    }

    std::vector<uint8_t> used(total_tokens, 0);
    std::mt19937 gen(42); // Fixed seed for reproducibility

    const size_t min_cluster_len = 4;
    const size_t max_cluster_len = 64;
    const size_t base_cluster_len = std::min(
        max_cluster_len,
        std::max(min_cluster_len, static_cast<size_t>(
            std::ceil(std::sqrt(static_cast<double>(num_selected)))) / 2));
    std::uniform_int_distribution<size_t> cluster_len_dist(
        min_cluster_len,
        std::max(min_cluster_len, base_cluster_len));

    while (indices.size() < num_selected) {
        const size_t remaining = num_selected - indices.size();
        const size_t cluster_len = std::min(remaining, cluster_len_dist(gen));
        const size_t max_start = total_tokens > cluster_len ? (total_tokens - cluster_len) : 0;
        std::uniform_int_distribution<size_t> start_dist(0, max_start);
        const size_t start = start_dist(gen);

        for (size_t offset = 0; offset < cluster_len && indices.size() < num_selected; ++offset) {
            const size_t idx = start + offset;
            if (idx >= total_tokens || used[idx]) {
                continue;
            }
            used[idx] = 1;
            indices.push_back(idx);
        }
    }

    std::sort(indices.begin(), indices.end());
    return indices;
}

std::vector<std::vector<size_t>> build_per_head_addresses(
    const std::vector<size_t>& token_indices,
    size_t context_length,
    bool randomize_order) {
    std::vector<size_t> ordered_indices = token_indices;
    if (randomize_order) {
        std::mt19937 gen(2026);
        std::shuffle(ordered_indices.begin(), ordered_indices.end(), gen);
    }

    const size_t bytes_per_head = context_length * TOKEN_SIZE_BYTES + HEAD_STRIDE_PADDING_BYTES;
    std::vector<std::vector<size_t>> per_head_addresses(NUM_HEAD_STREAMS);

    for (size_t head = 0; head < NUM_HEAD_STREAMS; ++head) {
        auto& head_addresses = per_head_addresses[head];
        head_addresses.reserve(ordered_indices.size());

        const size_t head_base = head * bytes_per_head;
        for (size_t token_index : ordered_indices) {
            head_addresses.push_back(head_base + token_index * TOKEN_SIZE_BYTES);
        }
    }

    return per_head_addresses;
}

// =================================================================================
// SGU Logic (Sparse Gather Unit)
// =================================================================================
std::vector<MemoryRequest> build_request_stream(
    const std::vector<size_t>& addresses,
    size_t max_merge_bytes) {
    std::vector<MemoryRequest> reqs;
    reqs.reserve(addresses.size());

    if (max_merge_bytes <= TOKEN_SIZE_BYTES) {
        size_t id_counter = 0;
        for (size_t address : addresses) {
            reqs.emplace_back(
                id_counter++,
                address,
                TOKEN_SIZE_BYTES,
                RequestType::READ,
                0
            );
        }
        return reqs;
    }

    if (addresses.empty()) return reqs;

    size_t id_counter = 0;
    size_t current_start_addr = addresses[0];
    size_t current_size = TOKEN_SIZE_BYTES;

    for (size_t i = 1; i < addresses.size(); ++i) {
        if (addresses[i] == addresses[i - 1] + TOKEN_SIZE_BYTES &&
            current_size + TOKEN_SIZE_BYTES <= max_merge_bytes) {
            current_size += TOKEN_SIZE_BYTES;
            continue;
        }
        reqs.emplace_back(id_counter++, current_start_addr, current_size, RequestType::READ, 0);
        current_start_addr = addresses[i];
        current_size = TOKEN_SIZE_BYTES;
    }
    reqs.emplace_back(id_counter++, current_start_addr, current_size, RequestType::READ, 0);

    return reqs;
}

std::vector<MemoryRequest> interleave_request_streams(
    const std::vector<std::vector<MemoryRequest>>& request_streams) {
    size_t total_requests = 0;
    for (const auto& stream : request_streams) {
        total_requests += stream.size();
    }

    std::vector<MemoryRequest> merged;
    merged.reserve(total_requests);
    std::vector<size_t> offsets(request_streams.size(), 0);
    size_t next_request_id = 0;

    bool has_progress = true;
    while (has_progress) {
        has_progress = false;
        for (size_t stream_id = 0; stream_id < request_streams.size(); ++stream_id) {
            if (offsets[stream_id] >= request_streams[stream_id].size()) {
                continue;
            }

            const auto& request = request_streams[stream_id][offsets[stream_id]++];
            merged.emplace_back(
                next_request_id++,
                request.address,
                request.size_bytes,
                request.type,
                0
            );
            has_progress = true;
        }
    }

    return merged;
}

// =================================================================================
// Simulation Engine
// =================================================================================
struct SimResult {
    size_t total_cycles;
    double effective_bandwidth_gbps;
    size_t total_bytes_transferred;
    size_t request_count;
    DDRController::Stats controller_stats;
};

size_t request_burst_count(const MemoryRequest& request) {
    constexpr size_t kCacheLineBytes = 64;
    return std::max<size_t>(1, (request.size_bytes + kCacheLineBytes - 1) / kCacheLineBytes);
}

SimResult run_simulation(
    DDRController& ddr,
    std::vector<MemoryRequest>& workload,
    size_t max_in_flight_bursts) {
    size_t current_cycle = 0;
    size_t req_idx = 0;
    size_t completed_count = 0;
    size_t total_reqs = workload.size();
    size_t total_bytes = 0;

    // Model finite request-buffer capacity in cache-line units so large merged
    // SGU bursts do not get unfairly throttled relative to 256B requests.
    size_t in_flight_bursts = 0;

    // Pre-calculate total bytes
    for (const auto& r : workload) total_bytes += r.size_bytes;

    while (completed_count < total_reqs) {
        // 1. Issue Requests (if bandwidth allows and buffer not full)
        while (req_idx < total_reqs) {
            const size_t request_bursts = request_burst_count(workload[req_idx]);
            if (in_flight_bursts + request_bursts > max_in_flight_bursts) {
                break;
            }

            workload[req_idx].arrival_cycle = current_cycle;
            if (ddr.send_request(workload[req_idx])) {
                in_flight_bursts += request_bursts;
                req_idx++;
            } else {
                // Controller full (backpressure)
                break;
            }
        }

        // 2. Step Memory System
        ddr.step(current_cycle);

        // 3. Collect Completions
        auto done = ddr.pop_completed_requests();
        if (!done.empty()) {
            completed_count += done.size();
            for (const auto& request : done) {
                in_flight_bursts -= request_burst_count(request);
            }
        }

        current_cycle++;
        
        // Safety Break (if stuck)
        if (current_cycle > 100000000) {
            std::cerr << "Simulation Timeout!" << std::endl;
            break;
        }
    }

    // Bandwidth = Bytes / (Cycles * Time_Per_Cycle)
    // Assume 200 MHz -> 5ns period
    double time_ns = current_cycle * 5.0;
    double gbps = (double)total_bytes / time_ns; // (Bytes / ns) is coincidentally GB/s

    return {current_cycle, gbps, total_bytes, total_reqs, ddr.get_stats()};
}

struct MethodConfig {
    const char* label;
    BankMappingStrategy mapping_strategy;
    size_t max_merge_bytes;
    bool use_hash_aware_scheduler;
    size_t max_in_flight_bursts;
};

struct RequestPlacement {
    uint32_t channel = 0;
    uint32_t bank = 0;
};

RequestPlacement estimate_request_placement(size_t address, BankMappingStrategy mapping_strategy) {
    RequestPlacement placement;
    const size_t block_addr = address >> 6;
    const uint32_t channel_bits = static_cast<uint32_t>(std::log2(DDR_CONFIG.num_channels));
    const uint32_t bank_bits = static_cast<uint32_t>(std::log2(DDR_CONFIG.num_banks_per_channel));
    const size_t blocks_per_row = std::max<size_t>(1, DDR_CONFIG.row_size_bytes >> 6);
    const uint32_t col_bits = static_cast<uint32_t>(std::log2(blocks_per_row));
    const size_t row = block_addr >> (channel_bits + bank_bits + col_bits);

    const uint32_t linear_channel =
        static_cast<uint32_t>(block_addr & (DDR_CONFIG.num_channels - 1));
    const uint32_t linear_bank = static_cast<uint32_t>(
        (block_addr >> channel_bits) & (DDR_CONFIG.num_banks_per_channel - 1));
    if (mapping_strategy == BankMappingStrategy::LINEAR) {
        placement.channel = linear_channel;
        placement.bank = linear_bank;
        return placement;
    }

    const size_t row_mix = row ^ (row >> channel_bits) ^ (row >> (channel_bits + bank_bits));
    const uint32_t row_channel_hash =
        static_cast<uint32_t>(row_mix) & (DDR_CONFIG.num_channels - 1);
    const uint32_t row_bank_hash = static_cast<uint32_t>(row_mix >> channel_bits) &
                                   (DDR_CONFIG.num_banks_per_channel - 1);

    placement.channel = (linear_channel ^ row_channel_hash) & (DDR_CONFIG.num_channels - 1);
    placement.bank = (linear_bank ^ row_bank_hash ^ row_channel_hash) &
                     (DDR_CONFIG.num_banks_per_channel - 1);
    return placement;
}

std::vector<MemoryRequest> rebalance_requests_by_channel_bank(
    const std::vector<MemoryRequest>& requests,
    BankMappingStrategy mapping_strategy,
    bool bank_only_buckets) {
    const size_t num_buckets = bank_only_buckets
                                   ? DDR_CONFIG.num_banks_per_channel
                                   : static_cast<size_t>(DDR_CONFIG.num_channels) *
                                         DDR_CONFIG.num_banks_per_channel;
    std::vector<std::vector<MemoryRequest>> buckets(num_buckets);
    for (const auto& request : requests) {
        const auto placement = estimate_request_placement(request.address, mapping_strategy);
        const size_t bucket_index = bank_only_buckets
                                        ? placement.bank
                                        : static_cast<size_t>(placement.channel) *
                                              DDR_CONFIG.num_banks_per_channel +
                                              placement.bank;
        buckets[bucket_index].push_back(request);
    }

    for (auto& bucket : buckets) {
        std::sort(bucket.begin(), bucket.end(), [](const MemoryRequest& lhs, const MemoryRequest& rhs) {
            if (lhs.address != rhs.address) return lhs.address < rhs.address;
            return lhs.size_bytes < rhs.size_bytes;
        });
    }

    std::vector<MemoryRequest> reordered;
    reordered.reserve(requests.size());
    std::vector<size_t> offsets(buckets.size(), 0);
    size_t next_request_id = 0;
    const size_t bucket_chunk = bank_only_buckets ? LARGE_HASH_BUCKET_CHUNK : HASH_BUCKET_CHUNK;
    bool has_progress = true;
    while (has_progress) {
        has_progress = false;
        for (size_t bucket = 0; bucket < buckets.size(); ++bucket) {
            for (size_t n = 0;
                 n < bucket_chunk && offsets[bucket] < buckets[bucket].size();
                 ++n) {
                const auto& request = buckets[bucket][offsets[bucket]++];
                reordered.emplace_back(
                    next_request_id++,
                    request.address,
                    request.size_bytes,
                    request.type,
                    0
                );
                has_progress = true;
            }
        }
    }

    return reordered;
}

std::vector<MemoryRequest> build_workload(
    const std::vector<size_t>& indices,
    size_t context_length,
    BankMappingStrategy mapping_strategy,
    size_t max_merge_bytes,
    bool use_hash_aware_scheduler) {
    const auto per_head_addresses = build_per_head_addresses(
        indices,
        context_length,
        max_merge_bytes <= TOKEN_SIZE_BYTES);

    std::vector<std::vector<MemoryRequest>> request_streams;
    request_streams.reserve(per_head_addresses.size());
    for (const auto& addresses : per_head_addresses) {
        request_streams.push_back(build_request_stream(addresses, max_merge_bytes));
    }

    auto workload = interleave_request_streams(request_streams);
    if (use_hash_aware_scheduler && mapping_strategy == BankMappingStrategy::HASH_XOR) {
        const bool use_bank_only_buckets =
            context_length >= LARGE_CONTEXT_BANK_ONLY_THRESHOLD;
        return rebalance_requests_by_channel_bank(
            workload,
            mapping_strategy,
            use_bank_only_buckets);
    }
    return workload;
}

std::string format_context_label(size_t total_tokens) {
    return std::to_string(total_tokens / 1024) + "K";
}

std::string format_percent(double value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(0) << value * 100.0 << "%";
    return oss.str();
}

std::string format_ratio(double ratio) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << ratio << "x";
    return oss.str();
}

struct DerivedStats {
    double row_hit_rate_pct = 0.0;
    double row_conflict_rate_pct = 0.0;
    double channel_skew = 0.0;
    double bank_skew = 0.0;
};

DerivedStats summarize_stats(const DDRController::Stats& stats) {
    DerivedStats derived;

    const size_t total_row_events = stats.row_hits + stats.row_conflicts + stats.row_misses;
    if (total_row_events > 0) {
        derived.row_hit_rate_pct = 100.0 * stats.row_hits / total_row_events;
        derived.row_conflict_rate_pct = 100.0 * stats.row_conflicts / total_row_events;
    }

    size_t total_channel_accesses = 0;
    size_t active_channels = 0;
    size_t max_channel_accesses = 0;
    for (size_t accesses : stats.channel_accesses) {
        total_channel_accesses += accesses;
        max_channel_accesses = std::max(max_channel_accesses, accesses);
        if (accesses > 0) {
            active_channels++;
        }
    }
    if (active_channels > 0) {
        const double avg_channel_accesses =
            static_cast<double>(total_channel_accesses) / active_channels;
        derived.channel_skew = max_channel_accesses / avg_channel_accesses;
    }

    size_t total_bank_accesses = 0;
    size_t active_banks = 0;
    size_t max_bank_accesses = 0;
    for (size_t accesses : stats.bank_accesses) {
        total_bank_accesses += accesses;
        max_bank_accesses = std::max(max_bank_accesses, accesses);
        if (accesses > 0) {
            active_banks++;
        }
    }
    if (active_banks > 0) {
        const double avg_bank_accesses =
            static_cast<double>(total_bank_accesses) / active_banks;
        derived.bank_skew = max_bank_accesses / avg_bank_accesses;
    }

    return derived;
}

// =================================================================================
// Main Experiment
// =================================================================================
int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << " Experiment 2: Memory Efficiency vs. Context Length " << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "Fixed selection ratio: " << format_percent(FIXED_SELECTION_RATIO)
              << " (effective sparsity "
              << format_percent(1.0 - FIXED_SELECTION_RATIO)
              << ") | Token Size: 256B | DDR Channels: 8 | Head Streams: "
              << NUM_HEAD_STREAMS << std::endl;
    std::cout << std::left << std::setw(10) << "Context"
              << std::setw(12) << "Sel/Head"
              << std::setw(18) << "Method"
              << std::setw(12) << "Requests"
              << std::setw(15) << "Cycles" 
              << std::setw(15) << "BW (GB/s)" 
              << std::setw(15) << "Speedup"
              << std::setw(12) << "Hit%"
              << std::setw(12) << "Conf%"
              << std::setw(12) << "ChSkew" << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------" << std::endl;

    const std::vector<size_t> context_lengths = {
        16 * 1024,
        32 * 1024,
        64 * 1024,
        128 * 1024,
        256 * 1024,
        512 * 1024,
    };
    const std::vector<MethodConfig> methods = {
        {"Baseline",  BankMappingStrategy::LINEAR,   TOKEN_SIZE_BYTES,   false, DEFAULT_MAX_IN_FLIGHT_BURSTS},
        {"Hash",      BankMappingStrategy::HASH_XOR, TOKEN_SIZE_BYTES,   false, DEFAULT_MAX_IN_FLIGHT_BURSTS},
        {"SGU",       BankMappingStrategy::LINEAR,   SGU_MERGE_BYTES,    false, DEFAULT_MAX_IN_FLIGHT_BURSTS},
        {"Hash+SGU",  BankMappingStrategy::HASH_XOR, HASH_SGU_MERGE_BYTES, true, HASH_SCHED_MAX_IN_FLIGHT_BURSTS},
    };

    for (size_t context_length : context_lengths) {
        // Generate Workload Indices (Shared)
        auto indices = generate_indices(FIXED_SELECTION_RATIO, context_length);

        size_t baseline_cycles = 0;
        bool first_row = true;

        for (const auto& method : methods) {
            DDRController::Config cfg = DDR_CONFIG;
            cfg.mapping_strategy = method.mapping_strategy;

            DDRController ddr(cfg);
            auto workload = build_workload(
                indices,
                context_length,
                method.mapping_strategy,
                method.max_merge_bytes,
                method.use_hash_aware_scheduler);
            SimResult result = run_simulation(ddr, workload, method.max_in_flight_bursts);
            DerivedStats derived = summarize_stats(result.controller_stats);

            if (baseline_cycles == 0) {
                baseline_cycles = result.total_cycles;
            }

            std::cout << std::left << std::setw(10)
                      << (first_row ? format_context_label(context_length) : "")
                      << std::setw(12)
                      << (first_row ? std::to_string(indices.size()) : "")
                      << std::setw(18) << method.label
                      << std::setw(12) << result.request_count
                      << std::setw(15) << result.total_cycles
                      << std::setw(15) << std::fixed << std::setprecision(2)
                      << result.effective_bandwidth_gbps
                      << std::setw(15)
                      << format_ratio((double)baseline_cycles / result.total_cycles)
                      << std::setw(12) << std::fixed << std::setprecision(1)
                      << derived.row_hit_rate_pct
                      << std::setw(12) << std::fixed << std::setprecision(1)
                      << derived.row_conflict_rate_pct
                      << std::setw(12) << std::fixed << std::setprecision(2)
                      << derived.channel_skew
                      << std::endl;
            first_row = false;
        }
        std::cout << "---------------------------------------------------------------------------------------------" << std::endl;
    }

    return 0;
}
