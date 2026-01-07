#include "PE/units/AddUnit.h"
#include "bf16/add_sim.h"
#include "bf16/bf16_basic_ops.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <memory>

void test_pipeline_addition() {
    std::cout << "Testing BF16 addition pipeline..." << std::endl;
    PE::AddUnit unit(std::make_unique<bf16::BF16AddPipeline>());

    uint16_t a_bf16 = bf16::float_to_bf16(2.5f);
    uint16_t b_bf16 = bf16::float_to_bf16(3.5f);

    unit.load_operands(PE::Number(a_bf16), PE::Number(b_bf16), true);
    unit.clock_cycle();

    for(int i = 0; i < 5; ++i) { // flush the pipeline
        unit.clock_cycle();
    }
    
    assert(unit.has_output());

    PE::Number add_res = unit.get_result();
    assert(std::abs(bf16::bf16_to_float(add_res.as_uint16()) - 6.0f) < 0.01);

    std::cout << "Pipeline addition test passed." << std::endl;
}

int main() {
    test_pipeline_addition();
    std::cout << "All addition pipeline tests passed!" << std::endl;
    return 0;
}