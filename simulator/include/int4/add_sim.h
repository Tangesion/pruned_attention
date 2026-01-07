#pragma once

#include "PE/base/VirtualPipeline.h"
#include "PE/base/PipelineInput.h"
#include "int4/int4_basic_ops.h"
#include <cstdint>
#include <vector>

namespace int4 {

class Int4AddPipeline : public PE::VirtualPipeline<int32_t> {
public:
    Int4AddPipeline(uint32_t latency = 1)
        : PE::VirtualPipeline<int32_t>(latency, [](const PE::PipelineInput& input) -> std::vector<int32_t> {
            const auto* in = dynamic_cast<const PE::TwoOperandInput32*>(&input);
            if (!in || !in->valid) return {};
            
            return { in->a + in->b };
        }) {}
};

} // namespace int4