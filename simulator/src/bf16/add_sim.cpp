#include "bf16/add_sim.h"
#include "PE/base/PipelineInput.h"
#include <cstdint>

namespace bf16 {

BF16AddPipeline::BF16AddPipeline() {
    reset();
}

void BF16AddPipeline::reset() {
    stage1 = {};
    stage2 = {};
    stage3 = {};
    stage4 = {};
    outputs.clear();
    cycle_count = 0;
}

bool BF16AddPipeline::is_active() const {
    return stage1.is_valid || stage2.is_valid || stage3.is_valid || stage4.is_valid;
}

void BF16AddPipeline::decompose_bf16(uint16_t bf16, uint16_t& sign, uint16_t& exponent, uint16_t& mantissa) {
    sign = (bf16 >> 15) & 0x1;
    exponent = (bf16 >> 7) & 0xFF;
    mantissa = bf16 & 0x7F;
}

uint16_t BF16AddPipeline::compose_bf16(uint16_t sign, uint16_t exponent, uint16_t mantissa) {
    return (sign << 15) | (exponent << 7) | (mantissa & 0x7F);
}

bool BF16AddPipeline::check_special_cases(uint16_t a, uint16_t b, uint16_t& result) {
    uint16_t sign_a, exp_a, mant_a;
    decompose_bf16(a, sign_a, exp_a, mant_a);
    uint16_t sign_b, exp_b, mant_b;
    decompose_bf16(b, sign_b, exp_b, mant_b);

    if ((exp_a == 0xFF && mant_a != 0) || (exp_b == 0xFF && mant_b != 0)) {
        result = NAN_VAL;
        return true;
    }

    if (exp_a == 0xFF) {
        if (exp_b == 0xFF && sign_a != sign_b) {
            result = NAN_VAL;
            return true;
        }
        result = compose_bf16(sign_a, 0xFF, 0);
        return true;
    }

    if (exp_b == 0xFF) {
        result = compose_bf16(sign_b, 0xFF, 0);
        return true;
    }

    if ((exp_a == 0 && mant_a == 0) && (exp_b == 0 && mant_b == 0)) {
        result = compose_bf16(sign_a == sign_b ? sign_a : 0, 0, 0);
        return true;
    }

    if (exp_a == 0 && mant_a == 0) {
        result = b;
        return true;
    }

    if (exp_b == 0 && mant_b == 0) {
        result = a;
        return true;
    }

    return false;
}

void BF16AddPipeline::clock_cycle(const PE::PipelineInput& input) {
    cycle_count++;

    const auto* two_op_input = dynamic_cast<const PE::TwoOperandInput*>(&input);
    bool valid = two_op_input && two_op_input->valid;
    uint16_t bf16_a = valid ? two_op_input->a : 0;
    uint16_t bf16_b = valid ? two_op_input->b : 0;

    // Stage 4: Normalization and Output (Previously Stage 5)
    stage4.is_valid = stage3.is_valid;
    if (stage3.is_valid) {
        uint16_t result_bf16 = 0;
        if (stage3.is_special) {
            result_bf16 = stage3.result;
        } else {
            uint16_t mant_result = stage3.mant_result;
            int exp_result = stage3.exp_result;
            uint16_t sign_result = stage3.sign_result;

            if (mant_result == 0) {
                result_bf16 = compose_bf16(0, 0, 0);
            } else {
                if (mant_result & 0x100) {
                    uint16_t round_bit = mant_result & 0x1;
                    mant_result >>= 1;
                    exp_result++;
                    if (round_bit && (mant_result & 0x1)) {
                        mant_result++;
                        if (mant_result & 0x100) {
                            mant_result >>= 1;
                            exp_result++;
                        }
                    }
                }

                while (mant_result && !(mant_result & 0x80)) {
                    mant_result <<= 1;
                    exp_result--;
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
    
    // Stage 3: Addition/Subtraction (Previously Stage 4)
    stage3.is_valid = stage2.is_valid;
    stage3.is_special = stage2.is_special;
    stage3.result = stage2.result;
    if (stage2.is_valid && !stage2.is_special) {
        if (stage2.sign_a == stage2.sign_b) {
            stage3.mant_result = stage2.mant_a + stage2.mant_b;
            stage3.sign_result = stage2.sign_a;
        } else {
            if (stage2.mant_a >= stage2.mant_b) {
                stage3.mant_result = stage2.mant_a - stage2.mant_b;
                stage3.sign_result = stage2.sign_a;
            } else {
                stage3.mant_result = stage2.mant_b - stage2.mant_a;
                stage3.sign_result = stage2.sign_b;
            }
        }
        stage3.exp_result = stage2.exp_result;
    }

    // Stage 2: Alignment (Previously Stage 3)
    stage2.is_valid = stage1.is_valid;
    stage2.is_special = stage1.is_special;
    stage2.result = stage1.result;
    if (stage1.is_valid && !stage1.is_special) {
        uint16_t exp_a = stage1.exp_a;
        uint16_t mant_a = stage1.mant_a;
        uint16_t exp_b = stage1.exp_b;
        uint16_t mant_b = stage1.mant_b;

        if (exp_a > exp_b) {
            int shift = exp_a - exp_b;
            mant_b = (shift > 24) ? 0 : (mant_b >> shift);
            stage2.exp_result = exp_a;
        } else if (exp_b > exp_a) {
            int shift = exp_b - exp_a;
            mant_a = (shift > 24) ? 0 : (mant_a >> shift);
            stage2.exp_result = exp_b;
        } else {
            stage2.exp_result = exp_a;
        }
        stage2.sign_a = stage1.sign_a;
        stage2.sign_b = stage1.sign_b;
        stage2.mant_a = mant_a;
        stage2.mant_b = mant_b;
    }
    
    // Stage 1: Input & Decomposition (Previously Stage 1 & 2)
    if (valid) {
        stage1.a = bf16_a;
        stage1.b = bf16_b;
        stage1.is_valid = true;
        stage1.is_special = check_special_cases(bf16_a, bf16_b, stage1.result);

        if (!stage1.is_special) {
             decompose_bf16(stage1.a, stage1.sign_a, stage1.exp_a, stage1.mant_a);
             decompose_bf16(stage1.b, stage1.sign_b, stage1.exp_b, stage1.mant_b);

             if (stage1.exp_a == 0 && stage1.mant_a != 0) {
                 int leading_bit = 0;
                 uint16_t temp_mant = stage1.mant_a;
                 while(temp_mant && !(temp_mant & 0x80)) {
                     temp_mant <<= 1;
                     leading_bit++;
                 }
                 stage1.exp_a = 1 - leading_bit;
                 stage1.mant_a <<= leading_bit;
             } else {
                 stage1.mant_a |= 0x80;
             }

             if (stage1.exp_b == 0 && stage1.mant_b != 0) {
                 int leading_bit = 0;
                 uint16_t temp_mant = stage1.mant_b;
                 while(temp_mant && !(temp_mant & 0x80)) {
                     temp_mant <<= 1;
                     leading_bit++;
                 }
                 stage1.exp_b = 1 - leading_bit;
                 stage1.mant_b <<= leading_bit;
             } else {
                 stage1.mant_b |= 0x80;
             }
        }
    } else {
        stage1 = {};
    }
}

const std::deque<uint16_t>& BF16AddPipeline::get_outputs() const {
    return outputs;
}

uint16_t BF16AddPipeline::pop_output() {
    if (outputs.empty()) {
        throw std::runtime_error("No outputs available to pop.");
    }
    uint16_t val = outputs.front();
    outputs.pop_front();
    return val;
}

} // namespace bf16