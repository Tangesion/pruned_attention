#include "PE/scheduler/GemvScheduler.h"
#include <iostream>

namespace PE {

GemvScheduler::GemvScheduler(std::unique_ptr<MacArrayUnit> unit, Config config)
    : ComputeScheduler(std::move(unit)), config(config) {}

MacArrayUnit* GemvScheduler::get_mac_array() {
    return static_cast<MacArrayUnit*>(hardware_resource.get());
}

std::vector<uint16_t> GemvScheduler::run_gemv(
    const std::vector<std::vector<uint16_t>>& matrix, 
    const std::vector<uint16_t>& input
) {
    if (matrix.empty() || input.empty()) return {};
    
    size_t total_rows = matrix.size();
    size_t cols = matrix[0].size();
    if (cols != input.size()) {
        // Simple error handling, could throw exception
        return {}; 
    }

    size_t num_pes = config.num_pes;
    MacArrayUnit* mac_array = get_mac_array();
    std::vector<uint16_t> final_results;
    final_results.reserve(total_rows);

    // Tiling loop: Process rows in batches of 'num_pes'
    for (size_t r = 0; r < total_rows; r += num_pes) {
        // Reset hardware for new batch to clear accumulators
        mac_array->reset();
        
        size_t current_batch_rows = std::min(num_pes, total_rows - r);
        
        // Streaming loop: Process columns
        for (size_t c = 0; c < cols; ++c) {
            // Simulate bandwidth latency for loading weights
            simulate_memory_latency(current_batch_rows);
            
            // Prepare inputs for this cycle
            std::vector<std::pair<uint16_t, uint16_t>> inputs(num_pes, {0, 0});
            std::vector<bool> valids(num_pes, false);
            
            for (size_t i = 0; i < current_batch_rows; ++i) {
                // Ensure we don't go out of bounds of the matrix
                if (r + i < total_rows) {
                    inputs[i] = {input[c], matrix[r+i][c]};
                    valids[i] = true;
                }
            }
            
            mac_array->load_inputs(inputs, valids);
            
            // Execute one clock cycle
            tick();
        }

        // Drain pipeline: Stop feeding and wait for completion
        std::vector<std::pair<uint16_t, uint16_t>> zeros(num_pes, {0, 0});
        std::vector<bool> invalids(num_pes, false);
        mac_array->load_inputs(zeros, invalids);
        
        while (mac_array->is_active()) {
            tick();
        }
        
        // Collect results
        // NOTE: This assumes pop_all_outputs returns one final result per unit
        // or a predictable sequence. This logic might need adjustment based on
        // MacArrayUnit implementation details.
        std::vector<uint16_t> batch_outputs = mac_array->pop_all_outputs();
        
        // Append valid results to final output
        for (size_t i = 0; i < current_batch_rows; ++i) {
             if (i < batch_outputs.size()) {
                 final_results.push_back(batch_outputs[i]);
             } else {
                 final_results.push_back(0); // Should not happen if logic is correct
             }
        }
    }
    
    return final_results;
}

void GemvScheduler::simulate_memory_latency(size_t words_count) {
    if (config.bandwidth_words_per_cycle <= 0) return;
    
    double cycles_needed = (double)words_count / config.bandwidth_words_per_cycle;
    int wait_cycles = static_cast<int>(std::ceil(cycles_needed)) - 1;
    if (wait_cycles < 0) wait_cycles = 0;
    
    MacArrayUnit* mac_array = get_mac_array();
    std::vector<std::pair<uint16_t, uint16_t>> zeros(config.num_pes, {0, 0});
    std::vector<bool> invalids(config.num_pes, false);

    for (int i = 0; i < wait_cycles; ++i) {
        mac_array->load_inputs(zeros, invalids);
        tick();
    }
}

} // namespace PE
