#include "PE/units/ReduceMacUnit.h"
#include "bf16/add_sim.h"
#include "bf16/multiply_sim.h"
#include "bf16/bf16_basic_ops.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>
#include <cmath>
#include <functional>
#include <cstdlib>

using namespace PE;

// --- Helpers ---
uint16_t f2bf(float f) {
    return bf16::float_to_bf16(f);
}

float bf2f(uint16_t b) {
    return bf16::bf16_to_float(b);
}

// Generic Test Runner
void run_gemv_test(
    std::string test_name,
    int K,
    int num_cols,
    std::function<std::pair<float, float>(int col, int k)> input_gen, // returns {in1, in2}
    std::function<float(int col)> expected_gen,                       // returns expected result for col
    bool insert_bubbles = false
) {
    std::cout << "\n[TEST] " << test_name << " (K=" << K << ", Cols=" << num_cols 
              << ", Bubbles=" << (insert_bubbles ? "ON" : "OFF") << ")" << std::endl;

    auto mult_factory = []() { return std::make_unique<bf16::BF16MultiplyPipeline>(); };
    auto add_factory = []() { return std::make_unique<bf16::BF16AddPipeline>(); };
    
    ReduceMacUnit reduce_unit(mult_factory, add_factory);
    reduce_unit.reset();

    int received_count = 0;
    int total_cycles = 0;
    int errors = 0;

    // Simulation Loop
    for (int col = 0; col < num_cols; ++col) {
        int k = 0;
        while (k < K) {
            total_cycles++;

            // Logic: Reset is asserted for the first 4 VALID inputs of a NEW column (col > 0)
            bool reset_flag = (col > 0) && (k < 4);
            
            auto [v1, v2] = input_gen(col, k);
            reduce_unit.load_inputs(f2bf(v1), f2bf(v2), true, reset_flag);
            reduce_unit.clock_cycle();
            k++; // Only increment processed data count on valid input

            // Check Output
            if (reduce_unit.has_output()) {
                float res = bf2f(reduce_unit.get_output());
                // The output received here is for the PREVIOUS column (col - 1)
                // because the current column's start triggers the flush.
                // However, strictly speaking, it's the output of the *just completed accumulation*.
                // The first output we get corresponds to Col 0.
                
                int result_idx = received_count;
                float expected = expected_gen(result_idx);
                
                if (std::abs(res - expected) > (std::abs(expected) * 0.05f + 0.5f)) {
                    std::cerr << "  ERROR at Cycle " << total_cycles << ": Col " << result_idx 
                              << " Expected " << expected << " Got " << res << std::endl;
                    errors++;
                } else {
                    // std::cout << "  [OK] Col " << result_idx << " Result: " << res << std::endl;
                }
                received_count++;
            }
        }
    }

    // Final Flush
    // std::cout << "  Final Flush..." << std::endl;
    for (int i = 0; i < ADD_PIPELINE_DEPTH; ++i) {
        total_cycles++;
        reduce_unit.load_inputs(0, 0, true, true);
        reduce_unit.clock_cycle();
    }
    

    for (int i = 0; reduce_unit.is_active(); ++i) { // Enough cycles to drain
        total_cycles++;
        reduce_unit.load_inputs(0, 0, false, false);
        reduce_unit.clock_cycle();

        if (reduce_unit.has_output()) {
             float res = bf2f(reduce_unit.get_output());
             int result_idx = received_count;
             if (result_idx < num_cols) {
                float expected = expected_gen(result_idx);
                if (std::abs(res - expected) > (std::abs(expected) * 0.05f + 0.5f)) {
                     std::cerr << "  ERROR (Flush) at Cycle " << total_cycles << ": Col " << result_idx 
                               << " Expected " << expected << " Got " << res << std::endl;
                     errors++;
                } else {
                     // std::cout << "  [OK] Col " << result_idx << " Result: " << res << std::endl;
                }
                received_count++;
             }
        }
    }

    if (received_count == num_cols && errors == 0) {
        std::cout << "  PASSED. Received " << received_count << " results." << std::endl;
    } else {
        std::cerr << "  FAILED. Received " << received_count << "/" << num_cols 
                  << " results. Errors: " << errors << std::endl;
        assert(false);
    }
}

// --- Specific Tests ---

void test_basic_ones() {
    run_gemv_test("Basic Ones", 32, 2, 
        [](int c, int k) { return std::make_pair(1.0f, 1.0f); },
        [](int c) { return 32.0f; } // 32 * 1 * 1
    );
}

void test_alternating() {
    // Inputs: 1, -1, 1, -1... 
    // Sum should be 0 (for even K)
    run_gemv_test("Alternating Signs (+1, -1)", 32, 4,
        [](int c, int k) { 
            float sign = (k % 2 == 0) ? 1.0f : -1.0f;
            return std::make_pair(1.0f, sign); 
        },
        [](int c) { return 0.0f; }
    );
}

void test_incremental() {
    // Col 0: 1*1 + 1*1... = K
    // Col 1: 2*1 + 2*1... = 2*K
    run_gemv_test("Incremental Column Values", 32, 3,
        [](int c, int k) { 
            float val = (float)(c + 1); 
            return std::make_pair(val, 1.0f); 
        },
        [](int c) { return 32.0f * (c + 1); }
    );
}

// ... (previous code)

void test_unaligned_padded() {
    // Real K=30, but we process 32 cycles per column (2 cycles padding)
    // Sum = 30.0
    run_gemv_test("Unaligned Padded (K=30 -> 32)", 32, 3,
        [](int c, int k) {
            if (k < 30) return std::make_pair(1.0f, 1.0f);
            return std::make_pair(0.0f, 0.0f); // Padding
        },
        [](int c) { return 30.0f; }
    );
}

void test_random() {
    int K = 2048;
    int cols = 5;
    run_gemv_test("Random Inputs", K, cols,
        [K](int c, int k) {
             // Simple deterministic pseudo-random
             float v = (float)((c * K + k) % 7 - 3); // range -3 to 3
             return std::make_pair(v, 1.0f);
        },
        [=](int c) {
             float sum = 0;
             for(int k=0; k<K; ++k) {
                 float v = (float)((c * K + k) % 7 - 3);
                 sum += v;
             }
             return sum;
        }
    );
}

int main() {
    std::srand(0); // Deterministic seed
    test_basic_ones();
    test_alternating();
    test_incremental();
    test_unaligned_padded();
    test_random();
    
    std::cout << "\nALL TESTS PASSED." << std::endl;
    return 0;
}
