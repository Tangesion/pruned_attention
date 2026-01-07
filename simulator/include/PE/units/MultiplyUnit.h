#pragma once

#include "PE/base/ComputeComponent.h"
#include "PE/base/Pipeline.h"
#include "PE/base/PipelineInput.h"
#include "PE/base/DataType.h"
#include "bf16/multiply_sim.h"
#include <memory>

namespace PE {

class MultiplyUnit : public ComputeComponent {
private:
    PipelinePtr<Number> pipeline;
    TwoOperandInput current_input;

public:
    explicit MultiplyUnit(PipelinePtr<Number> p);

    void reset() override;

    bool is_active() const override;

    void clock_cycle() override;

    void load_operands(Number a, Number b, bool valid);

    Number get_result();

    bool has_output() const;
};

} // namespace PE