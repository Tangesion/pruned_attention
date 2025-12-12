#include "bf16/add_sim.h"
#include "bf16/bf16_basic_ops.h"
#include <iostream>
#include <cassert>
#include <cmath>

void test_pipeline_addition() {
    std::cout << "Testing BF16 addition pipeline..." << std::endl;
    bf16::BF16AddPipeline pipeline;

    uint16_t a_bf16 = bf16::float_to_bf16(2.5f);
    uint16_t b_bf16 = bf16::float_to_bf16(3.5f);

    pipeline.clock_cycle(a_bf16, b_bf16, true);
    for(int i = 0; i < 5; ++i) { // flush the pipeline
        pipeline.clock_cycle(0, 0, false);
    }
    
    const auto& outputs = pipeline.get_outputs();
    assert(outputs.size() == 1);

    uint16_t add_res = outputs.front();
    assert(std::abs(bf16::bf16_to_float(add_res) - 6.0f) < 0.01);

    std::cout << "Pipeline addition test passed." << std::endl;
}

int main() {
    test_pipeline_addition();
    std::cout << "All addition pipeline tests passed!" << std::endl;
    return 0;
}