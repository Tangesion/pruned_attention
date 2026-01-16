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

// =================================================================================
// Shared Implementation (Single Batch)
// =================================================================================
std::vector<Number> GemvScheduler::run_gemv_impl(
    const std::vector<std::vector<uint16_t>>& matrix, 
    const std::vector<uint16_t>& input
) {
    if (matrix.empty()) return {};
    size_t num_pes = config.num_pes;
    size_t K = input.size();
    size_t N = matrix[0].size();
    std::vector<Number> output(N, Number());

    // Phase 1: Back-to-back Compute Phase
    for (size_t n_start = 0; n_start < N; n_start += num_pes) {
        size_t current_batch_size = std::min(num_pes, N - n_start);
        for (size_t k = 0; k < K; ++k) {
            std::vector<std::pair<Number, Number>> inputs;
            std::vector<bool> valids;
            std::vector<bool> reset_flags;
            bool reset_flag = (n_start > 0) && (k < 4);
            
            inputs.reserve(num_pes);
            valids.reserve(num_pes);
            reset_flags.reserve(num_pes);

            for (size_t p = 0; p < num_pes; ++p) {
                if (p < current_batch_size) {
                    inputs.emplace_back(Number(matrix[k][n_start + p]), Number(input[k]));
                    valids.push_back(true);
                    reset_flags.push_back(reset_flag);
                } else {
                    inputs.emplace_back(Number(), Number());
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
            std::vector<std::pair<Number, Number>> flush_in(num_pes, {Number(), Number()});
            std::vector<bool> flush_v(num_pes, true); 
            std::vector<bool> flush_r(num_pes, true);
            get_mac_array()->load_inputs(flush_in, flush_v, flush_r);
            clock_cycle();
        }
    }

    while (get_mac_array()->is_active()) {
        get_mac_array()->load_inputs(
            std::vector<std::pair<Number, Number>>(num_pes, {Number(), Number()}),
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

// =================================================================================
// Shared Implementation (Multi Batch)
// =================================================================================
std::vector<std::vector<Number>> GemvScheduler::run_gemv_impl_multibatch(
    const std::vector<std::vector<std::vector<uint16_t>>>& matrices, 
    const std::vector<std::vector<uint16_t>>& inputs
) {
    std::vector<std::vector<Number>> outputs;
    if (matrices.empty() || inputs.empty()) return outputs;

    size_t B = matrices.size();
    size_t K = inputs[0].size();
    // Assume all batches have the same N
    size_t N = matrices[0].empty() ? 0 : matrices[0][0].size(); 
    size_t num_pes = config.num_pes;

    // Initialize outputs
    outputs.resize(B, std::vector<Number>(N, Number()));

    // Calculate how many batches can fit in parallel
    // If N > num_pes, max_parallel_batches will be 0
    size_t max_parallel_batches = (N > 0) ? (num_pes / N) : 0;

    // Case 1: Matrix width N is larger than total PEs.
    if (max_parallel_batches == 0) {
        for (size_t b = 0; b < B; ++b) {
            hardware_component->reset();
            std::vector<Number> batch_res = run_gemv_impl(matrices[b], inputs[b]);
            outputs[b] = batch_res;
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
            std::vector<std::pair<Number, Number>> packed_inputs;
            std::vector<bool> valids;
            std::vector<bool> reset_flags;
            
            packed_inputs.reserve(num_pes);
            valids.reserve(num_pes);
            reset_flags.reserve(num_pes);

            for (size_t p = 0; p < num_pes; ++p) {
                // Map PE ID -> Local Batch ID + Column ID
                size_t local_b_idx = p / N; 
                size_t n_idx = p % N;

                if (local_b_idx < current_chunk_B) {
                    size_t global_b_idx = b_start + local_b_idx;
                    packed_inputs.emplace_back(Number(matrices[global_b_idx][k][n_idx]), Number(inputs[global_b_idx][k]));
                    valids.push_back(true);
                    reset_flags.push_back(false); 
                } else {
                    packed_inputs.emplace_back(Number(), Number());
                    valids.push_back(false);
                    reset_flags.push_back(false);
                }
            }
            get_mac_array()->load_inputs(packed_inputs, valids, reset_flags);
            clock_cycle();
        }

        // 3. Flush & Drain Phase
        for (size_t i = 0; i < ADD_PIPELINE_DEPTH; i++) {
             std::vector<std::pair<Number, Number>> flush_in(num_pes, {Number(), Number()});
             std::vector<bool> flush_v(num_pes, true); 
             std::vector<bool> flush_r(num_pes, true);
             get_mac_array()->load_inputs(flush_in, flush_v, flush_r);
             clock_cycle();
        }

        while (get_mac_array()->is_active()) {
            get_mac_array()->load_inputs(
                std::vector<std::pair<Number, Number>>(num_pes, {Number(), Number()}),
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
                if (get_mac_array()->has_output(p)) {
                    get_mac_array()->pop_output(p);
                }
            }
        }
    }

    return outputs;
}

// =================================================================================
// Public Wrappers
// =================================================================================

// BF16 Single Batch
std::vector<uint16_t> GemvScheduler::run_gemv(
    const std::vector<std::vector<uint16_t>>& matrix, 
    const std::vector<uint16_t>& input
) {
    auto res_num = run_gemv_impl(matrix, input);
    std::vector<uint16_t> res(res_num.size());
    for(size_t i=0; i<res_num.size(); ++i) res[i] = res_num[i].as_uint16();
    return res;
}

// BF16 Multi Batch
std::vector<std::vector<uint16_t>> GemvScheduler::run_gemv(
    const std::vector<std::vector<std::vector<uint16_t>>>& matrices, 
    const std::vector<std::vector<uint16_t>>& inputs
) {
    auto res_num = run_gemv_impl_multibatch(matrices, inputs);
    std::vector<std::vector<uint16_t>> res(res_num.size());
    for(size_t i=0; i<res_num.size(); ++i) {
        res[i].resize(res_num[i].size());
        for(size_t j=0; j<res_num[i].size(); ++j) {
            res[i][j] = res_num[i][j].as_uint16();
        }
    }
    return res;
}

// Int4 Single Batch
std::vector<int32_t> GemvScheduler::run_gemv_int32(
    const std::vector<std::vector<uint16_t>>& matrix, 
    const std::vector<uint16_t>& input
) {
    auto res_num = run_gemv_impl(matrix, input);
    std::vector<int32_t> res(res_num.size());
    for(size_t i=0; i<res_num.size(); ++i) res[i] = res_num[i].as_int32();
    return res;
}

// Int4 Multi Batch
std::vector<std::vector<int32_t>> GemvScheduler::run_gemv_int32(
    const std::vector<std::vector<std::vector<uint16_t>>>& matrices, 
    const std::vector<std::vector<uint16_t>>& inputs
) {
    auto res_num = run_gemv_impl_multibatch(matrices, inputs);
    std::vector<std::vector<int32_t>> res(res_num.size());
    for(size_t i=0; i<res_num.size(); ++i) {
        res[i].resize(res_num[i].size());
        for(size_t j=0; j<res_num[i].size(); ++j) {
            res[i][j] = res_num[i][j].as_int32();
        }
    }
    return res;
}

} // namespace PE
