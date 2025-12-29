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

std::vector<std::vector<uint16_t>> GemvScheduler::run_gemv(
    const std::vector<std::vector<std::vector<uint16_t>>>& matrices, 
    const std::vector<std::vector<uint16_t>>& inputs
) {
    std::vector<std::vector<uint16_t>> outputs;
    if (matrices.empty() || inputs.empty()) return outputs;

    size_t B = matrices.size();
    size_t K = inputs[0].size();
    // Assume all batches have the same N
    size_t N = matrices[0].empty() ? 0 : matrices[0][0].size(); 
    size_t num_pes = config.num_pes;

    // Initialize outputs
    outputs.resize(B, std::vector<uint16_t>(N, 0));

    // Calculate how many batches can fit in parallel
    // If N > num_pes, max_parallel_batches will be 0
    size_t max_parallel_batches = (N > 0) ? (num_pes / N) : 0;

    // Case 1: Matrix width N is larger than total PEs.

    if (max_parallel_batches == 0) {
        for (size_t b = 0; b < B; ++b) {
            hardware_component->reset();
            outputs[b] = run_gemv(matrices[b], inputs[b]);
        }
        return outputs;
    }

    // Case 2: Can fit at least 1 batch (and possibly more) in parallel.

    size_t chunk_size = max_parallel_batches;

    for (size_t b_start = 0; b_start < B; b_start += chunk_size) {
        size_t current_chunk_B = std::min(chunk_size, B - b_start);
        
        // 1. Reset hardware for the new chunk of parallel tasks
        hardware_component->reset();

        // 2. Compute Phase
        for (size_t k = 0; k < K; ++k) {
            std::vector<std::pair<uint16_t, uint16_t>> packed_inputs;
            std::vector<bool> valids;
            std::vector<bool> reset_flags;
            
            packed_inputs.reserve(num_pes);
            valids.reserve(num_pes);
            reset_flags.reserve(num_pes);

            for (size_t p = 0; p < num_pes; ++p) {
                // Map PE ID -> Local Batch ID + Column ID
                // p = 0..N-1       -> local_b=0
                // p = N..2N-1      -> local_b=1
                size_t local_b_idx = p / N; 
                size_t n_idx = p % N;

                if (local_b_idx < current_chunk_B) {
                    size_t global_b_idx = b_start + local_b_idx;
                    
                    packed_inputs.emplace_back(matrices[global_b_idx][k][n_idx], inputs[global_b_idx][k]);
                    valids.push_back(true);
                    // Global reset at start of chunk is sufficient
                    reset_flags.push_back(false); 
                } else {
                    // Padding for unused PEs (e.g., remaining PEs in the last chunk or incomplete blocks)
                    packed_inputs.emplace_back(0, 0);
                    valids.push_back(false);
                    reset_flags.push_back(false);
                }
            }
            get_mac_array()->load_inputs(packed_inputs, valids, reset_flags);
            clock_cycle();
        }

        // 3. Flush & Drain Phase
        for (size_t i = 0; i < ADD_PIPELINE_DEPTH; i++) {
             std::vector<std::pair<uint16_t, uint16_t>> flush_in(num_pes, {0, 0});
             std::vector<bool> flush_v(num_pes, true); 
             std::vector<bool> flush_r(num_pes, true);
             get_mac_array()->load_inputs(flush_in, flush_v, flush_r);
             clock_cycle();
        }

        while (get_mac_array()->is_active()) {
            get_mac_array()->load_inputs(
                std::vector<std::pair<uint16_t, uint16_t>>(num_pes, {0, 0}),
                std::vector<bool>(num_pes, false),
                std::vector<bool>(num_pes, false)
            );
            clock_cycle();
        }

        // 4. Collection Phase
        for (size_t p = 0; p < num_pes; ++p) {
            size_t local_b_idx = p / N;
            size_t n_idx = p % N;
            
            if (local_b_idx < current_chunk_B) {
                if (get_mac_array()->has_output(p)) {
                    size_t global_b_idx = b_start + local_b_idx;
                    outputs[global_b_idx][n_idx] = get_mac_array()->pop_output(p);
                }
            } else {
                // Drain any garbage/residual outputs from unused PEs if they exist
                if (get_mac_array()->has_output(p)) {
                    get_mac_array()->pop_output(p);
                }
            }
        }
    }

    return outputs;
}



} // namespace PE
