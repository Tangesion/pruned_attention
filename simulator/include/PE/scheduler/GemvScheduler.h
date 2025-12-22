#pragma once
#include "PE/scheduler/ComputeScheduler.h"
#include "PE/units/MacArrayUnit.h"
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

    GemvScheduler(std::unique_ptr<MacArrayUnit> unit, Config config);

    // Run GEMV task
    // matrix: [Rows x Cols]
    // input:  [Cols]
    // return: [Rows]
    std::vector<uint16_t> run_gemv(
        const std::vector<std::vector<uint16_t>>& matrix, 
        const std::vector<uint16_t>& input
    );

private:
    Config config;
    
    // Helper to simulate memory latency
    void simulate_memory_latency(size_t words_count);
    
    // Helper to access the component as MacArrayUnit
    MacArrayUnit* get_mac_array();
};

} // namespace PE
