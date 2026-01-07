#pragma once
#include <cstdint>
#include <vector>
#include "PE/base/DataType.h"

namespace PE {

// Base struct for all pipeline inputs
struct PipelineInput {
    virtual ~PipelineInput() = default;
    bool valid = false;
};

// Input struct for two-operand operations (using Generic Number)
struct TwoOperandInput : PipelineInput {
    Number a, b;
    bool valid = false;
};

// Input struct for fp32 conversion operations (kept as is for specific ops, or could use Number)
struct ConvertInput : PipelineInput {
    uint32_t val;
};

// Vector Input (Updated to use Number)
struct VectorInput : PipelineInput {
    std::vector<Number> values;
    bool valid = false;
    int param = 0;
};


} // namespace PE
