#pragma once

#include "PE/base/VirtualPipeline.h"
#include "PE/base/PipelineInput.h"
#include <cstdint>
#include <vector>

namespace int4 {

// Helper for sign extension
inline int32_t sign_extend_int4(uint16_t val) {
    // Take lower 4 bits
    uint8_t v = val & 0xF;
    // Check sign bit (bit 3)
    if (v & 0x8) {
        return static_cast<int32_t>(v) | 0xFFFFFFF0;
    } else {
        return static_cast<int32_t>(v);
    }
}

// Input for MAC operation with 32-bit accumulator
struct Int4MacInput : public PE::PipelineInput {
    uint16_t a;
    uint16_t b;
    int32_t c; // Accumulator
};

class Int4MultiplyPipeline : public PE::VirtualPipeline<int32_t> {
public:
    Int4MultiplyPipeline(uint32_t latency = 1) 
        : PE::VirtualPipeline<int32_t>(latency, [](const PE::PipelineInput& input) -> std::vector<int32_t> {
            const auto* in = dynamic_cast<const PE::TwoOperandInput*>(&input);
            if (!in || !in->valid) return {};
            
            int32_t op_a = sign_extend_int4(in->a);
            int32_t op_b = sign_extend_int4(in->b);
            
            return { op_a * op_b };
        }) {}
};

class Int4MacPipeline : public PE::VirtualPipeline<int32_t> {
public:
    Int4MacPipeline(uint32_t latency = 1) 
        : PE::VirtualPipeline<int32_t>(latency, [](const PE::PipelineInput& input) -> std::vector<int32_t> {
            const auto* in = dynamic_cast<const Int4MacInput*>(&input);
            if (!in || !in->valid) return {};
            
            int32_t op_a = sign_extend_int4(in->a);
            int32_t op_b = sign_extend_int4(in->b);
            int32_t acc = in->c;
            
            return { acc + (op_a * op_b) };
        }) {}
};

} // namespace int4
