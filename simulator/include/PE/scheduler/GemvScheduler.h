#pragma once
#include "PE/scheduler/ComputeScheduler.h"
#include "PE/units/ReduceMacArrayUnit.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace PE {

class GemvScheduler : public ComputeScheduler {
public:
    struct Config {
        double bandwidth_words_per_cycle; // Bandwidth configuration
        size_t num_pes;                   // Number of PEs in the MacArrayUnit
    };

    GemvScheduler(std::unique_ptr<ReduceMacArrayUnit> unit, Config config);

    // Run single batch GEMV task
    // matrix: [K x N]
    // input:  [1, K]
    // return: [1, N]
    std::vector<uint16_t> run_gemv(
        const std::vector<std::vector<uint16_t>>& matrix, 
        const std::vector<uint16_t>& input
    );

    // Run multiple batches like mha GEMV task
    // matrices: [B x K x N]
    // inputs:   [B x 1 x K]
    // return:   [B x 1 x N]
    std::vector<std::vector<uint16_t>> run_gemv(
        const std::vector<std::vector<std::vector<uint16_t>>>& matrices, 
        const std::vector<std::vector<uint16_t>>& inputs
    );

    // Run multiple batches like multi-batches input single matrix GEMV task
    // matrix: [K x N]
    // inputs: [B x 1 x K]
    // return: [B x 1 x N]
    std::vector<std::vector<uint16_t>> run_gemv(
        const std::vector<std::vector<uint16_t>>& matrix, 
        const std::vector<std::vector<uint16_t>>& inputs
    );


private:
    Config config;
    
    // Helper to simulate memory latency
    void simulate_memory_latency(size_t words_count);
    
    // Helper to access the component as MacArrayUnit
    ReduceMacArrayUnit* get_mac_array();
};

} // namespace PE
