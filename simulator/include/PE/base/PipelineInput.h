#pragma once
#include <cstdint>
#include <vector>

namespace PE {

// Base struct for all pipeline inputs
struct PipelineInput {
    virtual ~PipelineInput() = default;
    bool valid = false;
};

// Input struct for two-operand operations
struct TwoOperandInput : PipelineInput {
    uint16_t a, b;
};

// Input struct for fp32 conversion operations
struct ConvertInput : PipelineInput {
    uint32_t val;
};

// A possible future input struct for three-operand operations
struct ThreeOperandInput : PipelineInput {
    uint16_t a, b, c;
    bool valid = false;
};

// Input vector
struct VectorInput : PipelineInput {
    std::vector<uint16_t> values;
    bool valid = false;
    int param = 0;
};


} // namespace PE
