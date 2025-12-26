#include "PE/scheduler/GemvScheduler.h"
#include "bf16/add_sim.h"
#include <cstdint>
#include <iostream>
#include <vector>

namespace PE {

GemvScheduler::GemvScheduler(std::unique_ptr<ReduceMacArrayUnit> unit, Config config)
    : ComputeScheduler(std::move(unit)), config(config) {
        reset();
    }

ReduceMacArrayUnit* GemvScheduler::get_mac_array() {
    return static_cast<ReduceMacArrayUnit*>(hardware_component.get());
}

std::vector<uint16_t> GemvScheduler::run_gemv(
    const std::vector<std::vector<uint16_t>>& matrix, 
    const std::vector<uint16_t>& input
) {
    if (matrix.empty()) return {};
    size_t num_pes = config.num_pes;
    size_t K = input.size();
    size_t N = matrix[0].size();
    std::vector<uint16_t> output(N, 0);
    // Phase 1: Back-to-back Compute Phase
    for (size_t n_start = 0; n_start < N; n_start += num_pes) {
        size_t current_batch_size = std::min(num_pes, N - n_start);
        for (size_t k = 0; k < K; ++k) {
            std::vector<std::pair<uint16_t, uint16_t>> inputs;
            std::vector<bool> valids;
            std::vector<bool> reset_flags;
            bool reset_flag = (n_start > 0) && (k < 4);
            for (size_t p = 0; p < num_pes; ++p) {
                if (p < current_batch_size) {
                    inputs.emplace_back(matrix[k][n_start + p], input[k]);
                    valids.push_back(true);
                    reset_flags.push_back(reset_flag);
                } else {
                    inputs.emplace_back(0, 0);
                    valids.push_back(false);
                    reset_flags.push_back(false);
                }
            }
            get_mac_array()->load_inputs(inputs, valids, reset_flags);
            clock_cycle();
        }
    }
    // Phase 2: Final Flush & Drain Phase
    {
        for (size_t i = 0; i < ADD_PIPELINE_DEPTH; i++) {
            std::vector<std::pair<uint16_t, uint16_t>> flush_in(num_pes, {0, 0});
            std::vector<bool> flush_v(num_pes, true); 
            std::vector<bool> flush_r(num_pes, true);
            get_mac_array()->load_inputs(flush_in, flush_v, flush_r);
            clock_cycle();
        }
    }

    while (get_mac_array()->is_active()) {
    //for (int i = 0; i < 100; ++i){}
        get_mac_array()->load_inputs(
            std::vector<std::pair<uint16_t, uint16_t>>(num_pes, {0, 0}),
            std::vector<bool>(num_pes, false),
            std::vector<bool>(num_pes, false)
        );
        clock_cycle();
    }
    // Phase 3: Collection Phase
    for (size_t n_start = 0; n_start < N; n_start += num_pes) {
        size_t current_batch_size = std::min(num_pes, N - n_start);
        for (size_t p = 0; p < current_batch_size; ++p) {
            if (get_mac_array()->has_output(p)) {
                output[n_start + p] = get_mac_array()->pop_output(p);
            }
        }
    }
    return output; 

}



} // namespace PE
