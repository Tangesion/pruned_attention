#include "PE/utils.h"
#include <memory>

namespace pe {

MultiplyUnit::MultiplyUnit()
    :   bf16_a(0),
        bf16_b(0),
        output(0),
        input_valid(false),
        output_valid(false),
        pipeline(std::make_unique<bf16::BF16MultiplyPipeline>()) {}

bool MultiplyUnit::is_input_valid() const {
    return input_valid;
}

bool MultiplyUnit::is_output_valid() const {
    return output_valid;
}

void MultiplyUnit::set_inputs(uint16_t a, uint16_t b, bool valid) {
    bf16_a = a;
    bf16_b = b;
    input_valid = valid;
}

void MultiplyUnit::clock_cycle() {
    pipeline->clock_cycle(bf16_a, bf16_b, input_valid);
    output_valid = pipeline->is_output_valid();
    if (output_valid) {
        output = pipeline->pop_output();
    }
}

void MultiplyUnit::reset() {
    pipeline->reset();
    input_valid = false;
    output_valid = false;
    bf16_a = 0;
    bf16_b = 0;
    output = 0;
}

AddUnit::AddUnit()
    :   bf16_a(0),
        bf16_b(0),
        output(0),
        input_valid(false),
        output_valid(false),
        pipeline(std::make_unique<bf16::BF16AddPipeline>()) {}

bool AddUnit::is_input_valid() const {
    return input_valid;
}

bool AddUnit::is_output_valid() const {
    return output_valid;
}

void AddUnit::set_inputs(uint16_t a, uint16_t b, bool valid) {
    bf16_a = a;
    bf16_b = b;
    input_valid = valid;
}

void AddUnit::clock_cycle() {
    pipeline->clock_cycle(bf16_a, bf16_b, input_valid);
    output_valid = pipeline->is_active();
    if (output_valid) {
        output = pipeline->pop_output();
    }
}

void AddUnit::reset() {
    pipeline->reset();
    input_valid = false;
    output_valid = false;
    bf16_a = 0;
    bf16_b = 0;
    output = 0;
}

} // namespace pe