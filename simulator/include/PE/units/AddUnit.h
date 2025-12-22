#pragma once

#include "PE/base/ComputeComponent.h"
#include "PE/base/Pipeline.h"
#include "PE/base/PipelineInput.h"
#include "bf16/add_sim.h"
#include <memory>

namespace PE {

class AddUnit : public ComputeComponent {
private:
    PipelinePtr pipeline;
    TwoOperandInput current_input;

public:
    explicit AddUnit(PipelinePtr p);

    void reset() override;

    bool is_active() const override;

    void clock_cycle() override;

    void load_operands(uint16_t a, uint16_t b, bool valid);

    uint16_t get_result();

    bool has_output() const;
};

} // namespace PE