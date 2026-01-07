#include "bf16/convert_sim.h"
#include "PE/base/PipelineInput.h"
#include <cstdint>
#include <cstring>

namespace bf16 {

BF16ConvertPipeline::BF16ConvertPipeline() {
    reset();
}

void BF16ConvertPipeline::reset() {
    stage1 = {};
    stage2 = {};
    stage3 = {};
    outputs.clear();
    cycle_count = 0;
}

bool BF16ConvertPipeline::is_active() const {
    return stage1.is_valid || stage2.is_valid || stage3.is_valid;
}

void BF16ConvertPipeline::decompose_fp32(uint32_t fp32_val, uint16_t& sign, uint16_t& exponent, uint16_t& mantissa) {
    sign = (fp32_val >> 31) & 0x1;
    exponent = (fp32_val >> 23) & 0xFF;
    mantissa = fp32_val & 0x7FFFFF;
}

void BF16ConvertPipeline::clock_cycle(const PE::PipelineInput& input) {
    cycle_count++;

    const auto* convert_input = dynamic_cast<const PE::ConvertInput*>(&input);
    bool new_valid = convert_input && convert_input->valid;
    uint32_t new_fp32 = new_valid ? convert_input->val : 0;

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
        outputs.push_back(PE::Number(stage3.bf16_val));
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

const std::deque<PE::Number>& BF16ConvertPipeline::get_outputs() const {
    return outputs;
}

PE::Number BF16ConvertPipeline::pop_output() {
    if (outputs.empty()) {
        throw std::runtime_error("No outputs available to pop.");
    }
    PE::Number val = outputs.front();
    outputs.pop_front();
    return val;
}

} // namespace bf16