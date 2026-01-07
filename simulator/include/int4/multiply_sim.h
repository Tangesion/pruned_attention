#pragma once

#include "PE/base/VirtualPipeline.h"
#include "PE/base/PipelineInput.h"
#include "int4/int4_basic_ops.h"
#include <cstdint>
#include <vector>

namespace int4 {

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

} // namespace int4