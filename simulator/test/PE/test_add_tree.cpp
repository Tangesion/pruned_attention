#include "PE/units/AddTreeUnit.h"
#include "bf16/add_sim.h"
#include "bf16/bf16_basic_ops.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include <iomanip>
#include <queue>
#include <algorithm>

using namespace PE;

// Helper to generate random inputs
std::vector<float> generate_random_inputs(size_t size) {
    std::vector<float> inputs(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) {
        inputs[i] = dis(gen);
    }
    return inputs;
}

// Reference reduction sum
float reference_sum(const std::vector<float>& inputs) {
    float sum = 0.0f;
    for (float v : inputs) {
        sum += v;
    } 
    return sum;
}

void test_stream_add_tree(size_t input_size, size_t batch_count) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Testing Streaming AddTreeUnit (Size: " << input_size << ", Batches: " << batch_count << ")" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // 1. Setup
    auto factory = []() { return std::make_unique<bf16::BF16AddPipeline>(); };
    PE::AddTreeUnit add_tree(factory, input_size);
    add_tree.reset();

    // 2. Prepare Data
    std::vector<std::vector<float>> all_inputs_f;
    std::vector<std::vector<uint16_t>> all_inputs_bf16;
    std::vector<float> expected_sums;

    for (size_t b = 0; b < batch_count; ++b) {
        auto inputs_f = generate_random_inputs(input_size);
        std::vector<uint16_t> inputs_bf16(input_size);
        for (size_t i = 0; i < input_size; ++i) {
            inputs_bf16[i] = bf16::float_to_bf16(inputs_f[i]);
        }
        all_inputs_f.push_back(inputs_f);
        all_inputs_bf16.push_back(inputs_bf16);
        expected_sums.push_back(reference_sum(inputs_f));
    }

    // 3. Run Simulation (Streaming)
    std::cout << "Streaming inputs..." << std::endl;
    std::vector<float> received_sums;
    int received_count = 0;
    int cycle = 0;
    size_t batch_idx = 0;
    
    // We feed one batch per cycle until all batches are fed.
    // Then we continue clocking until all results are drained.
    
    while (received_count < batch_count && cycle < 1000) {
        cycle++;
        
        // Feed input if available
        if (batch_idx < batch_count) {
            // Convert to Number vector
            std::vector<PE::Number> inputs_num;
            inputs_num.reserve(input_size);
            for(auto v : all_inputs_bf16[batch_idx]) {
                inputs_num.push_back(PE::Number(v));
            }
            add_tree.load_inputs(inputs_num, true);
            batch_idx++;
        } else {
            // Feed bubbles (implicit, or valid=false if required)
            // Original code didn't call load_inputs here, assuming AddTreeUnit handles it?
            // Checking AddTreeUnit.cpp: if is_active(), logic runs.
            // But clock_cycle clears stage_outputs for next stage.
            // Inputs to first stage are pushed by load_inputs.
            // If load_inputs not called, stage_outputs[0] is not populated.
            // Wait, AddTreeUnit logic pops from stage_outputs[i].
            // If we don't call load_inputs, stage_outputs[0] is empty.
            // Then stage 0 adders get 0,0,valid=false?
            // "if (stage_valids[i] && !stage_outputs[i].empty())"
            // If not called, stage_valids[0] is stale? No, stage_valids is updated in loop?
            // "stage_valids[0] = valid" in load_inputs.
            // If load_inputs NOT called, stage_valids[0] remains what it was?
            // No, load_inputs sets it. If not called, it retains previous value?
            // This suggests load_inputs MUST be called every cycle to set valid=false if no input.
            // Or AddTreeUnit logic needs fix.
            // Assuming for now we must feed bubbles.
            std::vector<PE::Number> bubbles(input_size, PE::Number());
            add_tree.load_inputs(bubbles, false);
        }

        add_tree.clock_cycle();

        if (add_tree.has_output()) {
            uint16_t res = add_tree.get_output().as_uint16();
            received_sums.push_back(bf16::bf16_to_float(res));
            received_count++;
            // std::cout << "Received batch " << received_count << " at cycle " << cycle << std::endl;
        }
    }

    // 4. Verify
    if (received_count < batch_count) {
        std::cerr << "FAIL: Timeout! Received " << received_count << "/" << batch_count << " batches." << std::endl;
        assert(false);
    }

    std::cout << "Total Cycles: " << cycle << std::endl;
    
    bool passed = true;
    for (size_t i = 0; i < batch_count; ++i) {
        float expected = expected_sums[i];
        float received = received_sums[i];
        float abs_err = std::abs(received - expected);
        float rel_err = std::abs(expected) > 1e-5 ? abs_err / std::abs(expected) : abs_err;

        // BF16 error tolerance
        if (abs_err > 2.0f && rel_err > 0.1f) {
            std::cerr << "Batch " << i << " Mismatch! Exp: " << expected << ", Rec: " << received << std::endl;
            passed = false;
        }
    }

    if (passed) {
        std::cout << "PASSED." << std::endl;
    } else {
        std::cerr << "FAIL: Data mismatch." << std::endl;
        assert(false);
    }
}

int main() {
    test_stream_add_tree(8, 5);
    test_stream_add_tree(16, 10);
    return 0;
}