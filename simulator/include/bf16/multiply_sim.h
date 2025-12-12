#ifndef MULTIPLY_SIM_H
#define MULTIPLY_SIM_H

#include <cstdint>
#include <deque>
#include <stdexcept>

namespace bf16 {

class BF16MultiplyPipeline {
public:
    BF16MultiplyPipeline();
    void reset();
    void clock_cycle(uint16_t bf16_a, uint16_t bf16_b, bool valid);
    const std::deque<uint16_t>& get_outputs() const;
    uint16_t pop_output();
    bool is_active() const;
    bool is_output_valid() const;
    
private:
    struct MultiplyPipelineStage {
        bool is_valid = false;
        bool is_special = false;
        uint16_t result = 0;
        uint16_t a = 0;
        uint16_t b = 0;
        uint16_t sign_a = 0;
        uint16_t exp_a = 0;
        uint16_t mant_a = 0;
        uint16_t sign_b = 0;
        uint16_t exp_b = 0;
        uint16_t mant_b = 0;
        int exp_result = 0;
        uint16_t sign_result = 0;
        uint32_t mant_result = 0;
    };

    void decompose_bf16(uint16_t bf16, uint16_t& sign, uint16_t& exponent, uint16_t& mantissa);
    uint16_t compose_bf16(uint16_t sign, uint16_t exponent, uint16_t mantissa);
    bool check_special_cases(uint16_t a, uint16_t b, uint16_t& result);

    MultiplyPipelineStage stage1;
    MultiplyPipelineStage stage2;
    MultiplyPipelineStage stage3;
    MultiplyPipelineStage stage4;
    MultiplyPipelineStage stage5;

    const uint16_t POS_INF = 0x7F80;
    const uint16_t NEG_INF = 0xFF80;
    const uint16_t NAN_VAL = 0x7FC0;

    std::deque<uint16_t> outputs;
    uint32_t cycle_count;
};

} // namespace bf16

#endif // MULTIPLY_SIM_H
