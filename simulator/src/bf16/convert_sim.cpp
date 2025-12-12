#include "bf16/convert_sim.h"
#include <cstdint>
#include <cstring> // For struct packing/unpacking simulation

namespace bf16 {

FP32toBF16Pipeline::FP32toBF16Pipeline() {
    reset();
}

void FP32toBF16Pipeline::reset() {
    stage1 = {};
    stage2 = {};
    stage3 = {};
    outputs.clear();
    cycle_count = 0;
}

void FP32toBF16Pipeline::decompose_fp32(uint32_t fp32_val, uint16_t& sign, uint16_t& exponent, uint16_t& mantissa) {
    sign = (fp32_val >> 31) & 0x1;
    exponent = (fp32_val >> 23) & 0xFF;
    mantissa = fp32_val & 0x7FFFFF;
}

void FP32toBF16Pipeline::clock_cycle(uint32_t new_fp32, bool new_valid) {
    cycle_count++;

    // Stage 3: Rounding
    if (stage2.is_valid) {
        if (stage2.exponent == 0xFF) {
            if (stage2.mantissa == 0) {
                stage3.bf16_val = stage2.high_bits;
            } else {
                stage3.bf16_val = (stage2.sign << 15) | (stage2.exponent << 7) | (stage2.mantissa >> 16) | 0x1;
            }
        } else {
            if (stage2.low_bits > 0x8000 || (stage2.low_bits == 0x8000 && (stage2.high_bits & 1))) {
                stage3.bf16_val = (stage2.high_bits + 1) & 0xFFFF;
            } else {
                stage3.bf16_val = stage2.high_bits;
            }
        }
        outputs.push_back(stage3.bf16_val);
    }
    stage3.is_valid = stage2.is_valid;

    // Stage 2: Decomposition
    stage2.fp32_val = stage1.fp32_val;
    stage2.high_bits = (stage1.fp32_val >> 16) & 0xFFFF;
    stage2.low_bits = stage1.fp32_val & 0xFFFF;
    decompose_fp32(stage1.fp32_val, stage2.sign, stage2.exponent, stage2.mantissa);
    stage2.is_valid = stage1.is_valid;

    // Stage 1: Input
    if (new_valid) {
        stage1.fp32_val = new_fp32;
    } else {
        stage1.fp32_val = 0;
    }
    stage1.is_valid = new_valid;
}

const std::deque<uint16_t>& FP32toBF16Pipeline::get_outputs() const {
    return outputs;
}

uint16_t FP32toBF16Pipeline::pop_output() {
    if (outputs.empty()) {
        throw std::runtime_error("No outputs available to pop.");
    }
    uint16_t val = outputs.front();
    outputs.pop_front();
    return val;
}

} // namespace bf16
