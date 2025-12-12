#ifndef CONVERT_SIM_H
#define CONVERT_SIM_H

#include <cstdint>
#include <deque>
#include <stdexcept>

namespace bf16 {

struct PipelineStage {
    uint32_t fp32_val;
    bool is_valid;
    uint16_t high_bits;
    uint16_t low_bits;
    uint16_t sign;
    uint16_t exponent;
    uint16_t mantissa;
    uint16_t bf16_val;
};

class FP32toBF16Pipeline {
public:
    FP32toBF16Pipeline();
    void reset();
    void clock_cycle(uint32_t new_fp32, bool new_valid);
    const std::deque<uint16_t>& get_outputs() const;
    uint16_t pop_output();

private:
    void decompose_fp32(uint32_t fp32_val, uint16_t& sign, uint16_t& exponent, uint16_t& mantissa);

    PipelineStage stage1;
    PipelineStage stage2;
    PipelineStage stage3;
    
    std::deque<uint16_t> outputs;
    uint32_t cycle_count;
};

} // namespace bf16

#endif // CONVERT_SIM_H
