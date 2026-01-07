#pragma once

#include "PE/base/ComputeComponent.h"
#include "PE/base/Pipeline.h"
#include "PE/base/PipelineInput.h" // For TwoOperandInput
#include "PE/base/DataType.h"
#include "bf16/add_sim.h"
#include <memory>

namespace PE {

class AddUnit : public ComputeComponent {
private:
    PipelinePtr<Number> pipeline;
    TwoOperandInput current_input; // Now contains Number

public:
    explicit AddUnit(PipelinePtr<Number> p);

    void reset() override;

    bool is_active() const override;

    void clock_cycle() override;

    // Load operands as Number
    void load_operands(Number a, Number b, bool valid);

    // Get result as Number
    Number get_result();

    bool has_output() const;
};

} // namespace PE