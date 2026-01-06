#ifndef CONVERT_SIM_H
#define CONVERT_SIM_H

#include "PE/base/Pipeline.h"
#include <cstdint>
#include <deque>
#include <stdexcept>

namespace bf16 {

struct ConvertPipelineStage {
    uint32_t fp32_val;
    bool is_valid;
    uint16_t high_bits;
    uint16_t low_bits;
    uint16_t sign;
    uint16_t exponent;
    uint16_t mantissa;
    uint16_t bf16_val;
};

class BF16ConvertPipeline final : public PE::Pipeline<uint16_t> {
public:
    BF16ConvertPipeline();
    void reset() override;
    void clock_cycle(const PE::PipelineInput& input) override;
    const std::deque<uint16_t>& get_outputs() const override;
    uint16_t pop_output() override;
    bool is_active() const override;

private:
    void decompose_fp32(uint32_t fp32_val, uint16_t& sign, uint16_t& exponent, uint16_t& mantissa);

    ConvertPipelineStage stage1;
    ConvertPipelineStage stage2;
    ConvertPipelineStage stage3;
};

} // namespace bf16

#endif // CONVERT_SIM_H
