#include "PE/units/ReduceMacUnit.h"
#include "bf16/add_sim.h"
#include "bf16/multiply_sim.h"
#include "bf16/bf16_basic_ops.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>
#include <cmath>

using namespace PE;

void test_stream_reduce_mac(int batch_count) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Testing Streaming ReduceMacUnit (Batches: " << batch_count << ")" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // 1. Setup
    auto mult_factory = []() { return std::make_unique<bf16::BF16MultiplyPipeline>(); };
    auto add_factory = []() { return std::make_unique<bf16::BF16AddPipeline>(); };
    
    ReduceMacUnit reduce_unit(mult_factory, add_factory);
    reduce_unit.reset();

    // 2. Configuration
    size_t input_len = 128; // 32 per lane
    float expected_val = 128.0f;
    uint16_t one_bf16 = bf16::float_to_bf16(1.0f);
    uint16_t zero_bf16 = bf16::float_to_bf16(0.0f);

    std::cout << "Streaming " << batch_count << " batches back-to-back..." << std::endl;

    int received_count = 0;
    int cycle = 0;
    int current_input_idx = 0;
    int current_batch = 0;
    
    // State machine for feeding:
    // 0: Feeding Inputs
    // 1: Feeding Reset (1 cycle)
    // 2: Done feeding
    int feed_state = 0; 

    // Loop until we collect all results or timeout
    while (received_count < batch_count && cycle < 10000) {
        cycle++;

        // --- Feeding Logic ---
        if (feed_state == 0) {
            // Feeding standard inputs
            if (current_batch < batch_count) {
                reduce_unit.load_inputs(one_bf16, one_bf16, true, false);
                current_input_idx++;
                if (current_input_idx == input_len) {
                    feed_state = 1; // Move to reset
                    current_input_idx = 0;
                }
            } else {
                feed_state = 2; // All batches fed
            }
        } else if (feed_state == 1) {
            // Feeding Reset Signal
            // We need 4 Resets to flush 4 lanes
            reduce_unit.load_inputs(zero_bf16, zero_bf16, true, true);
            current_input_idx++;
            if (current_input_idx == 4) { // 4 Resets
                feed_state = 0; // Back to inputs for next batch
                current_batch++;
                current_input_idx = 0;
            }
        } else {
            // Draining (Feed bubbles)
            reduce_unit.load_inputs(zero_bf16, zero_bf16, false, false);
        }

        reduce_unit.clock_cycle();

        // --- Collection Logic ---
        if (reduce_unit.has_output()) {
            uint16_t res = reduce_unit.get_output();
            float val = bf16::bf16_to_float(res);
            received_count++;
            
            // Verify immediately
            if (std::abs(val - expected_val) > 1.0f) {
                std::cerr << "Mismatch at Batch " << received_count << ": Expected " << expected_val << ", Got " << val << std::endl;
                assert(false);
            }
            // std::cout << "Received Batch " << received_count << " Result: " << val << " at Cycle " << cycle << std::endl;
        }
    }

    if (received_count < batch_count) {
        std::cerr << "Timeout! Received " << received_count << "/" << batch_count << std::endl;
        assert(false);
    }

    std::cout << "Total Cycles: " << cycle << std::endl;
    // Theoretical cycles:
    // Per Batch: 128 (Input) + 4 (Reset) = 132 cycles.
    // Latency: Mac Output (Buffer) + AddTree (8) = ~12 cycles lag.
    // Total ~= 132 * Batches + Latency.
    int expected_cycles = (128 + 4) * batch_count + 12; 
    std::cout << "Theoretical Min Cycles: ~" << expected_cycles << std::endl; 
    
    std::cout << "PASSED." << std::endl;
}

int main() {
    test_stream_reduce_mac(5);
    test_stream_reduce_mac(10);
    return 0;
}