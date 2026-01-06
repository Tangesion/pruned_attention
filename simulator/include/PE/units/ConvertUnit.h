#pragma once

#include "PE/base/ComputeComponent.h"
#include "PE/base/Pipeline.h"
#include "PE/base/PipelineInput.h"
#include "bf16/convert_sim.h"
#include <memory>

namespace PE {

class ConvertUnit : public ComputeComponent {
private:
    PipelinePtr<uint16_t> pipeline;
    ConvertInput current_input;

public:
    explicit ConvertUnit(PipelinePtr<uint16_t> p);

    void reset() override;

    bool is_active() const override;

    void clock_cycle() override;

    void load_operand(uint32_t val, bool valid);

    uint16_t get_result();

    bool has_output() const;
};

} // namespace PE