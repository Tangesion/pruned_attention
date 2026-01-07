#pragma once

#include "PE/base/VirtualPipeline.h"
#include "PE/base/PipelineInput.h"
#include "PE/base/DataType.h"
#include <cstdint>
#include <vector>

namespace int4 {

class Int4AddPipeline : public PE::VirtualPipeline<PE::Number> {
public:
    Int4AddPipeline(uint32_t latency = 1)
        : PE::VirtualPipeline<PE::Number>(latency, [](const PE::PipelineInput& input) -> std::vector<PE::Number> {
            const auto* in = dynamic_cast<const PE::TwoOperandInput*>(&input);
            if (!in || !in->valid) return {};
            
            // Assume inputs are Int32 for accumulation
            return { PE::Number(in->a.as_int32() + in->b.as_int32()) };
        }) {}
};

} // namespace int4
