#ifndef ADD_SIM_H
#define ADD_SIM_H

#include "PE/base/Pipeline.h"
#include <cstdint>
#include <deque>
#include <stdexcept>

#define ADD_PIPELINE_DEPTH 4

namespace bf16 {

class BF16AddPipeline final : public PE::Pipeline {
public:
    BF16AddPipeline();
    void reset() override;
    void clock_cycle(const PE::PipelineInput& input) override;
    const std::deque<uint16_t>& get_outputs() const override;
    uint16_t pop_output() override;
    bool is_active() const override;

private:
    struct AddPipelineStage {
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
        uint16_t mant_result = 0;
    };

    void decompose_bf16(uint16_t bf16, uint16_t& sign, uint16_t& exponent, uint16_t& mantissa);
    uint16_t compose_bf16(uint16_t sign, uint16_t exponent, uint16_t mantissa);
    bool check_special_cases(uint16_t a, uint16_t b, uint16_t& result);

    AddPipelineStage stage1;
    AddPipelineStage stage2;
    AddPipelineStage stage3;
    AddPipelineStage stage4;

    const uint16_t POS_INF = 0x7F80;
    const uint16_t NEG_INF = 0xFF80;
    const uint16_t NAN_VAL = 0x7FC0;
};

} // namespace bf16

#endif // ADD_SIM_H
