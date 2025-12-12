#include "PE/units/ConvertUnit.h"
#include "bf16/convert_sim.h"
#include "bf16/bf16_basic_ops.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>

void test_pipeline_conversion() {
    std::cout << "Testing FP32 to BF16 pipeline conversion..." << std::endl;
    PE::ConvertUnit unit(std::make_unique<bf16::BF16ConvertPipeline>());
    
    float test_val_f = 3.14159f;
    uint32_t test_val_u;
    memcpy(&test_val_u, &test_val_f, sizeof(test_val_f));

    unit.load_operand(test_val_u, true);
    unit.clock_cycle();
    unit.clock_cycle();
    unit.clock_cycle();
    unit.clock_cycle();

    assert(unit.has_output());

    uint16_t bf16_val = unit.get_result();
    float converted_val = bf16::bf16_to_float(bf16_val);
    assert(std::abs(test_val_f - converted_val) < 0.01);

    std::cout << "Pipeline conversion test passed." << std::endl;
}

int main() {
    test_pipeline_conversion();
    std::cout << "All conversion pipeline tests passed!" << std::endl;
    return 0;
}
