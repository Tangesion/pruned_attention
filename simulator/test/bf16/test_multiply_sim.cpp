#include "PE/units/MultiplyUnit.h"
#include "bf16/multiply_sim.h"
#include "bf16/bf16_basic_ops.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <memory>

void test_pipeline_multiplication() {
    std::cout << "Testing BF16 multiplication pipeline..." << std::endl;
    PE::MultiplyUnit unit(std::make_unique<bf16::BF16MultiplyPipeline>());

    uint16_t a_bf16 = bf16::float_to_bf16(2.5f);
    uint16_t b_bf16 = bf16::float_to_bf16(3.5f);

    unit.load_operands(a_bf16, b_bf16, true);
    unit.clock_cycle();

    for(int i = 0; i < 5; ++i) { // flush the pipeline
        unit.clock_cycle();
    }
    
    assert(unit.has_output());

    uint16_t mul_res = unit.get_result();
    assert(std::abs(bf16::bf16_to_float(mul_res) - 8.75f) < 0.01);

    std::cout << "Pipeline multiplication test passed." << std::endl;
}

int main() {
    test_pipeline_multiplication();
    std::cout << "All multiplication pipeline tests passed!" << std::endl;
    return 0;
}
