#pragma once

#include "PE/base/ComputeComponent.h"
#include "PE/base/Pipeline.h"
#include "PE/base/PipelineInput.h"
#include "PE/base/DataType.h"
#include "bf16/convert_sim.h"
#include <memory>

namespace PE {

class ConvertUnit : public ComputeComponent {
private:
    PipelinePtr<Number> pipeline;
    ConvertInput current_input;

public:
    explicit ConvertUnit(PipelinePtr<Number> p);

    void reset() override;

    bool is_active() const override;

    void clock_cycle() override;

    void load_operand(uint32_t val, bool valid);

    Number get_result();

    bool has_output() const;
};

} // namespace PE