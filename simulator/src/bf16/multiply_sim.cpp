#include "bf16/multiply_sim.h"
#include <cstdint>

namespace bf16 {

BF16MultiplyPipeline::BF16MultiplyPipeline() {
    reset();
}

void BF16MultiplyPipeline::reset() {
    stage1 = {};
    stage2 = {};
    stage3 = {};
    stage4 = {};
    stage5 = {};
    outputs.clear();
    cycle_count = 0;
}

bool BF16MultiplyPipeline::is_active() const {
    return stage1.is_valid || stage2.is_valid || stage3.is_valid || stage4.is_valid || stage5.is_valid;
}

void BF16MultiplyPipeline::decompose_bf16(uint16_t bf16, uint16_t& sign, uint16_t& exponent, uint16_t& mantissa) {
    sign = (bf16 >> 15) & 0x1;
    exponent = (bf16 >> 7) & 0xFF;
    mantissa = bf16 & 0x7F;
}

uint16_t BF16MultiplyPipeline::compose_bf16(uint16_t sign, uint16_t exponent, uint16_t mantissa) {
    return (sign << 15) | (exponent << 7) | (mantissa & 0x7F);
}

bool BF16MultiplyPipeline::check_special_cases(uint16_t a, uint16_t b, uint16_t& result) {
    uint16_t sign_a, exp_a, mant_a;
    decompose_bf16(a, sign_a, exp_a, mant_a);
    uint16_t sign_b, exp_b, mant_b;
    decompose_bf16(b, sign_b, exp_b, mant_b);

    uint16_t sign_result = sign_a ^ sign_b;

    if ((exp_a == 0xFF && mant_a != 0) || (exp_b == 0xFF && mant_b != 0)) {
        result = NAN_VAL;
        return true;
    }

    if ((exp_a == 0xFF && exp_b == 0 && mant_b == 0) || (exp_b == 0xFF && exp_a == 0 && mant_a == 0)) {
        result = NAN_VAL;
        return true;
    }

    if ((exp_a == 0 && mant_a == 0) || (exp_b == 0 && mant_b == 0)) {
        result = compose_bf16(sign_result, 0, 0);
        return true;
    }

    if (exp_a == 0xFF || exp_b == 0xFF) {
        result = compose_bf16(sign_result, 0xFF, 0);
        return true;
    }

    return false;
}

void BF16MultiplyPipeline::clock_cycle(uint16_t bf16_a, uint16_t bf16_b, bool valid) {
    cycle_count++;

    // Stage 5: Normalization and Output
    stage5.is_valid = stage4.is_valid;
    if (stage4.is_valid) {
        uint16_t result_bf16 = 0;
        if (stage4.is_special) {
            result_bf16 = stage4.result;
        } else {
            uint32_t mant_result = stage4.mant_result;
            int exp_result = stage4.exp_result;
            uint16_t sign_result = stage4.sign_result;

            if (mant_result == 0) {
                result_bf16 = compose_bf16(sign_result, 0, 0);
            } else {
                 if (mant_result & 0x8000) {
                    exp_result++;
                    uint16_t round_bits = mant_result & 0xFF;
                    uint16_t half_point = 0x80;
                    mant_result >>= 8;
                     if (round_bits > half_point || (round_bits == half_point && (mant_result & 0x1))) {
                        mant_result++;
                        if (mant_result & 0x100) {
                            mant_result >>= 1;
                            exp_result++;
                        }
                    }
                } else {
                    uint16_t round_bits = mant_result & 0x7F;
                    uint16_t half_point = 0x40; // Corrected half point for 7 bit mantissa
                    mant_result >>= 7;
                     if (round_bits > half_point || (round_bits == half_point && (mant_result & 0x1))) {
                        mant_result++;
                        if (mant_result & 0x100) {
                            mant_result >>= 1;
                            exp_result++;
                        }
                    }
                }

                mant_result &= 0x7F;

                if (exp_result <= 0) {
                    if (exp_result < -6) {
                        result_bf16 = compose_bf16(sign_result, 0, 0);
                    } else {
                        uint16_t denorm_mant = 0x80 | mant_result;
                        int shift_amount = 1 - exp_result;
                        uint16_t round_bit = (denorm_mant >> (shift_amount - 1)) & 1;
                        bool sticky_bits = (denorm_mant & ((1 << (shift_amount - 1)) - 1)) != 0;
                        denorm_mant >>= shift_amount;
                        if (round_bit && (sticky_bits || (denorm_mant & 1))) {
                            denorm_mant++;
                        }
                        result_bf16 = compose_bf16(sign_result, 0, denorm_mant & 0x7F);
                    }
                } else if (exp_result >= 0xFF) {
                    result_bf16 = compose_bf16(sign_result, 0xFF, 0);
                } else {
                    result_bf16 = compose_bf16(sign_result, exp_result, mant_result);
                }
            }
        }
        outputs.push_back(result_bf16);
    }

    // Stage 4: Rounding
    stage4.is_valid = stage3.is_valid;
    stage4.is_special = stage3.is_special;
    stage4.result = stage3.result;
    stage4.sign_result = stage3.sign_result;
    stage4.exp_result = stage3.exp_result;
    if (stage3.is_valid && !stage3.is_special) {
        stage4.mant_result = stage3.mant_result;
    }

    // Stage 3: Mantissa Multiplication
    stage3.is_valid = stage2.is_valid;
    stage3.is_special = stage2.is_special;
    stage3.result = stage2.result;
    stage3.sign_result = stage2.sign_result;
    if (stage2.is_valid && !stage2.is_special) {
        stage3.exp_result = stage2.exp_a + stage2.exp_b - 127;
        stage3.mant_result = static_cast<uint32_t>(stage2.mant_a) * stage2.mant_b;
    }

    // Stage 2: Decomposition
    stage2.is_valid = stage1.is_valid;
    stage2.is_special = stage1.is_special;
    stage2.result = stage1.result;
    if (stage1.is_valid && !stage1.is_special) {
        decompose_bf16(stage1.a, stage2.sign_a, stage2.exp_a, stage2.mant_a);
        decompose_bf16(stage1.b, stage2.sign_b, stage2.exp_b, stage2.mant_b);

        stage2.sign_result = stage2.sign_a ^ stage2.sign_b;

        if (stage2.exp_a == 0 && stage2.mant_a != 0) {
            int leading_bit = 0;
            uint16_t temp_mant = stage2.mant_a;
            while(temp_mant && !(temp_mant & 0x80)) {
                temp_mant <<= 1;
                leading_bit++;
            }
            stage2.exp_a = 1 - leading_bit;
            stage2.mant_a <<= leading_bit;
        } else {
            stage2.mant_a |= 0x80;
        }

        if (stage2.exp_b == 0 && stage2.mant_b != 0) {
            int leading_bit = 0;
            uint16_t temp_mant = stage2.mant_b;
            while(temp_mant && !(temp_mant & 0x80)) {
                temp_mant <<= 1;
                leading_bit++;
            }
            stage2.exp_b = 1 - leading_bit;
            stage2.mant_b <<= leading_bit;
        } else {
            stage2.mant_b |= 0x80;
        }
    }

    // Stage 1: Input
    if (valid) {
        stage1.a = bf16_a;
        stage1.b = bf16_b;
        stage1.is_valid = true;
        stage1.is_special = check_special_cases(bf16_a, bf16_b, stage1.result);
    } else {
        stage1 = {};
    }
}

const std::deque<uint16_t>& BF16MultiplyPipeline::get_outputs() const {
    return outputs;
}

uint16_t BF16MultiplyPipeline::pop_output() {
    if (outputs.empty()) {
        throw std::runtime_error("No outputs available to pop.");
    }
    uint16_t val = outputs.front();
    outputs.pop_front();
    return val;
}

bool BF16MultiplyPipeline::is_output_valid() const {
    return stage5.is_valid;
}

} // namespace bf16
