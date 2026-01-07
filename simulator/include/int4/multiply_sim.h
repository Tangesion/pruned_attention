#pragma once

#include "PE/base/VirtualPipeline.h"
#include "PE/base/PipelineInput.h"
#include "PE/base/DataType.h"
#include "int4/int4_basic_ops.h"
#include <cstdint>
#include <vector>

namespace int4 {

class Int4MultiplyPipeline : public PE::VirtualPipeline<PE::Number> {
public:
    Int4MultiplyPipeline(uint32_t latency = 1)
        : PE::VirtualPipeline<PE::Number>(latency, [](const PE::PipelineInput& input) -> std::vector<PE::Number> {
            const auto* in = dynamic_cast<const PE::TwoOperandInput*>(&input);
            if (!in || !in->valid) return {};
            
            // Unpack Int4 from uint16_t bits
            int32_t op_a = sign_extend_int4(in->a.as_uint16());
            int32_t op_b = sign_extend_int4(in->b.as_uint16());
            
            // Return as Int32 Number
            return { PE::Number(op_a * op_b) };
        }) {}
};

} // namespace int4
