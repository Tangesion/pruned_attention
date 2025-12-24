#include "PE/units/MacUnit.h"
#include "PE/units/AddUnit.h"
#include "PE/units/MultiplyUnit.h"
#include "bf16/add_sim.h"
#include "bf16/multiply_sim.h"
#include "bf16/bf16_basic_ops.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <memory>
#include <vector>
#include <utility>
#include <deque>
#include <iomanip>

struct TestInput {
    float a;
    float b;
    bool reset;
    bool valid;
};

void run_mac_test(const std::string& test_name, 
                 const std::vector<TestInput>& inputs, 
                 const std::vector<float>& expected_results) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Test: " << test_name << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // 1. Setup
    auto mult_pipe = std::make_unique<bf16::BF16MultiplyPipeline>();
    auto add_pipe = std::make_unique<bf16::BF16AddPipeline>();
    PE::MacUnit mac(std::move(mult_pipe), std::move(add_pipe));
    mac.reset();

    std::vector<float> received_results;

    // Helper to collect outputs
    auto collect = [&](PE::MacUnit& unit) {
        while (unit.has_final_output()) {
            received_results.push_back(bf16::bf16_to_float(unit.get_final_output()));
        }
    };

    // 2. Input Phase
    int cycle = 0;
    int actual_ops = 0;
    std::cout << "Phase 1: Feeding " << inputs.size() << " inputs..." << std::endl;
    for (const auto& in : inputs) {
        cycle++;
        if (in.valid) {
            mac.load_inputs(bf16::float_to_bf16(in.a), bf16::float_to_bf16(in.b), true, in.reset);
            // Count actual work (ignoring padding/dummy zeros if desired, 
            // but usually we count all valid cycles as 'ops' in hardware)
            // Here let's count only the inputs that weren't resets or paddings for a strict 'useful work' metric,
            // or just follow GemvScheduler's style of K*N.
            // For simplicity, let's count cycles where a valid multiplication was requested.
            actual_ops++;
        } else {
            mac.load_inputs(0, 0, false, false);
        }
        mac.clock_cycle();
        collect(mac);
    }

    // 3. Drain Phase
    std::cout << "Phase 2: Draining pipeline..." << std::endl;
    int drain_cycles = 0;
    // Provide bubbles until unit is idle
    while (mac.is_active()) {
        cycle++;
        drain_cycles++;
        mac.load_inputs(0, 0, false, false);
        mac.clock_cycle();
        collect(mac);
        
        if (drain_cycles > 100) {
            std::cerr << "Timeout draining pipeline!" << std::endl;
            break;
        }
    }
    
    double utilization = (double)actual_ops / cycle * 100.0;
    std::cout << "Stats:" << std::endl;
    std::cout << "  Total Cycles: " << cycle << " (Drain: " << drain_cycles << ")" << std::endl;
    std::cout << "  Actual Ops:   " << actual_ops << std::endl;
    std::cout << "  Utilization:  " << std::fixed << std::setprecision(2) << utilization << "%" << std::endl;

    // 4. Verification
    std::cout << "Expected: ";
    for(auto v : expected_results) std::cout << v << " ";
    std::cout << "\nReceived: ";
    for(auto v : received_results) std::cout << v << " ";
    std::cout << std::endl;

    assert(received_results.size() == expected_results.size());
    for(size_t i=0; i<expected_results.size(); ++i) {
        float diff = std::abs(received_results[i] - expected_results[i]);
        if (diff >= 0.1f) {
            std::cerr << "Mismatch at index " << i << ": " << received_results[i] << " != " << expected_results[i] << std::endl;
        }
        assert(diff < 0.1f);
    }
    std::cout << "PASSED." << std::endl;
}

void test_mac_unit_scenarios() {
    // Scenario 1: Aligned Data (8 items, multiple of 4)
    // Lane 0: 1*1 + 2*2 = 5
    // Lane 1: 1*2 + 2*3 = 8
    // Lane 2: 1*3 + 2*4 = 11
    // Lane 3: 1*4 + 2*5 = 14
    std::vector<TestInput> inputs_aligned;
    // Group A
    inputs_aligned.push_back({1.0f, 1.0f, false, true});
    inputs_aligned.push_back({1.0f, 2.0f, false, true});
    inputs_aligned.push_back({1.0f, 3.0f, false, true});
    inputs_aligned.push_back({1.0f, 4.0f, false, true});
    // Bubble
    inputs_aligned.push_back({0.0f, 0.0f, false, false});
    // Group B
    inputs_aligned.push_back({2.0f, 2.0f, false, true});
    inputs_aligned.push_back({2.0f, 3.0f, false, true});
    inputs_aligned.push_back({2.0f, 4.0f, false, true});
    inputs_aligned.push_back({2.0f, 5.0f, false, true});
    // 4 Resets
    for(int i=0; i<4; ++i) inputs_aligned.push_back({0.0f, 0.0f, true, true});

    run_mac_test("Aligned Data (8 items)", inputs_aligned, {5.0f, 8.0f, 11.0f, 14.0f});


    // Scenario 2: Non-Aligned Data (7 items) with Padding
    // Lane 0: 2, Lane 1: 2, Lane 2: 2, Lane 3: 1
    // Input ends at Lane 2. Next is Lane 3.
    // Padding: Insert 1 VALID ZERO (Lane 3). Next becomes Lane 0.
    // Resets: Reset L0, Reset L1, Reset L2, Reset L3.
    // Expected Output: L0(2), L1(2), L2(2), L3(1) -> {2.0, 2.0, 2.0, 1.0}
    std::vector<TestInput> inputs_7;
    for(int i=0; i<7; ++i) {
        inputs_7.push_back({1.0f, 1.0f, false, true});
    }
    // Pad to align to multiple of 4 (7 -> 8). MUST BE VALID to advance accumulator!
    inputs_7.push_back({0.0f, 0.0f, false, true}); // Valid Zero
    // 4 Resets
    for(int i=0; i<4; ++i) inputs_7.push_back({0.0f, 0.0f, true, true});

    run_mac_test("Non-Aligned Data (7 items) + Padding", inputs_7, {2.0f, 2.0f, 2.0f, 1.0f});


    // Scenario 3: Non-Aligned Data (10 items) with Padding
    // Lane 0: 3, Lane 1: 3, Lane 2: 2, Lane 3: 2
    // Input ends at Lane 1. Next is Lane 2.
    // Padding: Insert 2 VALID ZEROs (Lane 2, Lane 3). Next becomes Lane 0.
    // Resets: Reset L0, Reset L1, Reset L2, Reset L3.
    // Expected Output: L0(3), L1(3), L2(2), L3(2) -> {3.0, 3.0, 2.0, 2.0}
    std::vector<TestInput> inputs_10;
    for(int i=0; i<10; ++i) {
        inputs_10.push_back({1.0f, 1.0f, false, true});
    }
    // Pad to align to multiple of 4 (10 -> 12). MUST BE VALID!
    inputs_10.push_back({0.0f, 0.0f, false, true}); // Valid Zero L2
    inputs_10.push_back({0.0f, 0.0f, false, true}); // Valid Zero L3
    // 4 Resets
    for(int i=0; i<4; ++i) inputs_10.push_back({0.0f, 0.0f, true, true});

    run_mac_test("Non-Aligned Data (10 items) + Padding", inputs_10, {3.0f, 3.0f, 2.0f, 2.0f});

    // Scenario 4: High Load Benchmark (4096 items, Alternating)
    // We want each lane to sum to 0.0 to avoid BF16 saturation.
    // Stride is 4.
    // Previous pattern (+1, -1...) caused Lane 0 to see only +1, Lane 1 only -1.
    // New Pattern: 1, 1, 1, 1, -1, -1, -1, -1 ...
    // Lane 0 sees: In[0]=1, In[4]=-1, In[8]=1... -> Sum = 0.
    // Lane 1 sees: In[1]=1, In[5]=-1, In[9]=1... -> Sum = 0.
    std::vector<TestInput> inputs_bench;
    inputs_bench.reserve(4100);
    for(int i=0; i<4096; ++i) {
        // Blocks of 4 to ensure stride-4 lanes see alternating values
        float val = ((i / 4) % 2 == 0) ? 1.0f : -1.0f;
        inputs_bench.push_back({val, 1.0f, false, true});
    }
    // 4 Resets
    for(int i=0; i<4; ++i) inputs_bench.push_back({0.0f, 0.0f, true, true});

    run_mac_test("High Load Benchmark (4096 items, Alternating)", inputs_bench, {0.0f, 0.0f, 0.0f, 0.0f});

    // Scenario 5: Standard 128 items (Medium load)
    // 128 items -> 32 items per lane.
    // Each item is 1.0 * 1.0 = 1.0.
    // Expected Sum per lane = 32.0.
    // 32.0 is well within BF16's exact range (up to 256).
    std::vector<TestInput> inputs_128;
    for(int i=0; i<128; ++i) {
        inputs_128.push_back({1.0f, 1.0f, false, true});
    }
    // 4 Resets
    for(int i=0; i<4; ++i) inputs_128.push_back({0.0f, 0.0f, true, true});

    run_mac_test("Standard 128 items", inputs_128, {32.0f, 32.0f, 32.0f, 32.0f});
}

int main() {
    test_mac_unit_scenarios();
    return 0;
}
