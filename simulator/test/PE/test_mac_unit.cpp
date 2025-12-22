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

void test_dot_product() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Testing MacUnit for Vector Dot Product" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // 1. Setup the MacUnit and its dependencies
    PE::PipelinePtr mult_pipe = std::make_unique<bf16::BF16MultiplyPipeline>();
    PE::PipelinePtr add_pipe = std::make_unique<bf16::BF16AddPipeline>();
    PE::MacUnit my_mac(std::move(mult_pipe), std::move(add_pipe));

    // 2. Prepare input data for dot product: [1, 2, 3] · [4, 5, 6]
    std::vector<std::pair<float, float>> input_vectors = {
        {1.0f, 4.0f},
        {2.0f, 5.0f},
        {3.0f, 6.0f}
    };
    float expected_result = 32.0f; // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

    std::cout << "Calculating dot product of [1, 2, 3] and [4, 5, 6]" << std::endl;
    std::cout << "Expected result: " << expected_result << std::endl;
    
    // 3. Reset and initialize the MAC unit
    my_mac.reset();
    //my_mac.set_initial_acc(bf16::float_to_bf16(0.0f));

    // 4. Main test loop
    std::vector<uint16_t> final_outputs;
    size_t input_idx = 0;
    int cycle = 0;
    const int max_cycles = 100;

    while ((input_idx < input_vectors.size() || my_mac.is_active()) && cycle < max_cycles) {
        if (input_idx < input_vectors.size()) {
            uint16_t in1 = bf16::float_to_bf16(input_vectors[input_idx].first);
            uint16_t in2 = bf16::float_to_bf16(input_vectors[input_idx].second);
            if (input_idx == 1) {
                my_mac.load_inputs(in1, in2, true, true); // Reset accumulator on last input
            } else {
                my_mac.load_inputs(in1, in2, true);
            }
            input_idx++;
        } else {
            my_mac.load_inputs(0, 0, false);
        }

        my_mac.clock_cycle();
        
        if (my_mac.has_final_output()) {
            final_outputs.push_back(my_mac.get_final_output());
        }
        
        cycle++;
    }

    std::cout << "\nPipeline finished in " << cycle << " cycles." << std::endl;

    // 5. Assert the results
    assert(!final_outputs.empty());

    // The final result of the dot product is the last valid output produced.
    float final_result_f = bf16::bf16_to_float(final_outputs.back());
    float error = std::abs(final_result_f - expected_result);

    std::cout << "Final accumulated result: " << final_result_f << std::endl;
    std::cout << "Error: " << error << std::endl;

    assert(error < 0.01);
    
    // Optional: Check intermediate results
    assert(final_outputs.size() == 3);
    assert(std::abs(bf16::bf16_to_float(final_outputs[0]) - 4.0f) < 0.01);  // 1*4 + 0
    assert(std::abs(bf16::bf16_to_float(final_outputs[1]) - 14.0f) < 0.01); // 2*5 + 4
    assert(std::abs(bf16::bf16_to_float(final_outputs[2]) - 32.0f) < 0.01); // 3*6 + 14

    std::cout << "MacUnit dot product test passed." << std::endl;
}

int main() {
    test_dot_product();
    std::cout << "\nAll MacUnit tests passed!" << std::endl;
    return 0;
}
