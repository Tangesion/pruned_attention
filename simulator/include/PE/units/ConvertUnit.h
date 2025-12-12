#pragma once

#include "PE/base/ComputeComponent.h"
#include "PE/base/Pipeline.h"
#include "PE/base/PipelineInput.h"
#include "bf16/convert_sim.h"
#include <memory>
#include <stdexcept>

namespace PE {

class ConvertUnit : public ComputeComponent {
private:
    PipelinePtr pipeline;
    ConvertInput current_input;

public:
    explicit ConvertUnit(PipelinePtr p) : pipeline(std::move(p)) {
        if (!pipeline) {
            throw std::invalid_argument("ConvertUnit received a null pipeline.");
        }
    }

    void reset() override {
        pipeline->reset();
        current_input = {};
    }

    bool is_active() const override {
        return pipeline->is_active();
    }

    void clock_cycle() override {
        pipeline->clock_cycle(current_input);
        current_input.valid = false;
    }

    void load_operand(uint32_t val, bool valid) {
        current_input.val = val;
        current_input.valid = valid;
    }

    uint16_t get_result() {
        return pipeline->pop_output();
    }

    bool has_output() const {
        return !pipeline->get_outputs().empty();
    }
};

} // namespace PE
