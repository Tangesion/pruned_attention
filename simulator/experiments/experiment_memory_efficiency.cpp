#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <iomanip>
#include <set>
#include <unordered_set>
#include "Memory/DDRController.h"
#include "Memory/MemoryDefs.h"

using namespace Memory;

// =================================================================================
// Configuration Constants
// =================================================================================
const uint64_t TOTAL_TOKENS = 128 * 1024; // 128K Context Window
const uint64_t HIDDEN_SIZE = 128;         // 128 channels (Head Dimension)
const uint64_t BYTES_PER_ELEMENT = 2;     // BF16
const uint64_t TOKEN_SIZE_BYTES = HIDDEN_SIZE * BYTES_PER_ELEMENT; // 256 Bytes per token

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
std::vector<uint64_t> generate_indices(double sparsity, uint64_t total_tokens) {
    uint64_t num_selected = static_cast<uint64_t>(total_tokens * sparsity);
    std::vector<uint64_t> indices;
    indices.reserve(num_selected);

    // Use a set to ensure uniqueness, then convert to vector
    std::unordered_set<uint64_t> distinct_indices;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<uint64_t> dist(0, total_tokens - 1);

    while (distinct_indices.size() < num_selected) {
        distinct_indices.insert(dist(gen));
    }

    indices.assign(distinct_indices.begin(), distinct_indices.end());
    return indices;
}

// =================================================================================
// SGU Logic (Sparse Gather Unit)
// =================================================================================
std::vector<MemoryRequest> sgu_naive(const std::vector<uint64_t>& token_indices) {
    // Naive: Just send requests as they come (random order), no merging
    std::vector<MemoryRequest> reqs;
    reqs.reserve(token_indices.size());
    uint64_t id_counter = 0;

    for (uint64_t idx : token_indices) {
        reqs.emplace_back(
            id_counter++, 
            idx * TOKEN_SIZE_BYTES, // Address
            TOKEN_SIZE_BYTES,       // Size
            RequestType::READ, 
            0                       // Will be set during injection
        );
    }
    return reqs;
}

std::vector<MemoryRequest> sgu_optimized(std::vector<uint64_t> token_indices) {
    // Ours: 
    // 1. Sort indices (Simulating Reorder Buffer / ROB)
    // 2. Merge adjacent requests (Simulating Request Merging)
    
    std::sort(token_indices.begin(), token_indices.end());

    std::vector<MemoryRequest> reqs;
    uint64_t id_counter = 0;

    if (token_indices.empty()) return reqs;

    uint64_t current_start_idx = token_indices[0];
    uint64_t current_count = 1;

    for (size_t i = 1; i < token_indices.size(); ++i) {
        if (token_indices[i] == current_start_idx + current_count) {
            // Adjacent: Merge
            current_count++;
        } else {
            // Gap: Flush current burst
            reqs.emplace_back(
                id_counter++,
                current_start_idx * TOKEN_SIZE_BYTES,
                current_count * TOKEN_SIZE_BYTES,
                RequestType::READ,
                0
            );
            // Start new burst
            current_start_idx = token_indices[i];
            current_count = 1;
        }
    }
    // Flush last
    reqs.emplace_back(
        id_counter++,
        current_start_idx * TOKEN_SIZE_BYTES,
        current_count * TOKEN_SIZE_BYTES,
        RequestType::READ,
        0
    );

    return reqs;
}

// =================================================================================
// Simulation Engine
// =================================================================================
struct SimResult {
    uint64_t total_cycles;
    double effective_bandwidth_gbps;
    uint64_t total_bytes_transferred;
};

SimResult run_simulation(DDRController& ddr, std::vector<MemoryRequest>& workload) {
    uint64_t current_cycle = 0;
    size_t req_idx = 0;
    uint64_t completed_count = 0;
    size_t total_reqs = workload.size();
    uint64_t total_bytes = 0;

    // To simulate finite MSHR / Request Buffer, we limit in-flight requests
    // Let's say SGU can hold 64 pending requests
    const size_t MAX_IN_FLIGHT = 64;
    size_t in_flight = 0;

    // Pre-calculate total bytes
    for (const auto& r : workload) total_bytes += r.size_bytes;

    while (completed_count < total_reqs) {
        // 1. Issue Requests (if bandwidth allows and buffer not full)
        while (req_idx < total_reqs && in_flight < MAX_IN_FLIGHT) {
            workload[req_idx].arrival_cycle = current_cycle;
            if (ddr.send_request(workload[req_idx])) {
                req_idx++;
                in_flight++;
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
            in_flight -= done.size();
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

    return {current_cycle, gbps, total_bytes};
}

// =================================================================================
// Main Experiment
// =================================================================================
int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << " Experiment 2: Memory Efficiency (Linear vs. Hash+Merge) " << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "Context: 128K Tokens | Token Size: 256B | DDR Channels: 8" << std::endl;
    std::cout << std::left << std::setw(10) << "Sparsity" 
              << std::setw(15) << "Config" 
              << std::setw(15) << "Cycles" 
              << std::setw(15) << "BW (GB/s)" 
              << std::setw(15) << "Speedup" << std::endl;
    std::cout << "-----------------------------------------------------------------" << std::endl;

    std::vector<double> sparsities = {0.01, 0.05, 0.10}; // 1%, 5%, 10%

    for (double sp : sparsities) {
        // Generate Workload Indices (Shared)
        auto indices = generate_indices(sp, TOTAL_TOKENS);
        
        // --- Baseline: Linear Mapping + Naive SGU ---
        DDRController::Config cfg_naive = DDR_CONFIG;
        cfg_naive.mapping_strategy = BankMappingStrategy::LINEAR;
        DDRController ddr_naive(cfg_naive);
        auto workload_naive = sgu_naive(indices);
        SimResult res_naive = run_simulation(ddr_naive, workload_naive);

        // --- Ours: Hash Mapping + Optimized SGU ---
        DDRController::Config cfg_ours = DDR_CONFIG;
        cfg_ours.mapping_strategy = BankMappingStrategy::HASH_XOR;
        DDRController ddr_ours(cfg_ours);
        auto workload_ours = sgu_optimized(indices); // Sort + Merge
        SimResult res_ours = run_simulation(ddr_ours, workload_ours);

        // Print Baseline
        std::cout << std::left << std::setw(10) << (std::to_string((int)(sp*100)) + "%") 
                  << std::setw(15) << "Baseline" 
                  << std::setw(15) << res_naive.total_cycles 
                  << std::setw(15) << std::fixed << std::setprecision(2) << res_naive.effective_bandwidth_gbps 
                  << std::setw(15) << "1.00x" << std::endl;

        // Print Ours
        std::cout << std::left << std::setw(10) << "" 
                  << std::setw(15) << "Ours" 
                  << std::setw(15) << res_ours.total_cycles 
                  << std::setw(15) << std::fixed << std::setprecision(2) << res_ours.effective_bandwidth_gbps 
                  << std::setw(15) << std::fixed << std::setprecision(2) << (double)res_naive.total_cycles / res_ours.total_cycles << "x" << std::endl;
        std::cout << "-----------------------------------------------------------------" << std::endl;
    }

    return 0;
}
