#pragma once

#include "PE/base/ComputeComponent.h"
#include "PE/base/Pipeline.h"
#include "PE/base/PipelineInput.h"
#include "bf16/multiply_sim.h"
#include <memory>
#include <stdexcept>

namespace PE {

class MultiplyUnit : public ComputeComponent {
private:
    PipelinePtr pipeline;
    TwoOperandInput current_input;

public:
    explicit MultiplyUnit(PipelinePtr p) : pipeline(std::move(p)) {
        if (!pipeline) {
            throw std::invalid_argument("MultiplyUnit received a null pipeline.");
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

    void load_operands(uint16_t a, uint16_t b, bool valid) {
        current_input.a = a;
        current_input.b = b;
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
