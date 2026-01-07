#include "PE/units/MultiplyUnit.h"
#include "PE/base/backend.h"
#include <stdexcept>

namespace PE {

MultiplyUnit::MultiplyUnit(PipelinePtr<Number> p) : pipeline(std::move(p)) {
    if (!pipeline) {
        throw std::invalid_argument("MultiplyUnit received a null pipeline.");
    }
}

void MultiplyUnit::reset() {
    pipeline->reset();
    current_input = {};
}

bool MultiplyUnit::is_active() const {
    return pipeline->is_active();
}

void MultiplyUnit::clock_cycle() {
    pipeline->clock_cycle(current_input);
    current_input.valid = false;
}

void MultiplyUnit::load_operands(Number a, Number b, bool valid) {
    current_input.a = a;
    current_input.b = b;
    current_input.valid = valid;
}

Number MultiplyUnit::get_result() {
    return pipeline->pop_output();
}

bool MultiplyUnit::has_output() const {
    return !pipeline->get_outputs().empty();
}

class MultiplyUnitCreator : public Backend::Creator {
public:
    ComputeComponent* onCreate(std::vector<PipelinePtr<Number>>&& pipes) const override {
        if (pipes.size() != 1) {
            throw std::invalid_argument("MultiplyUnitCreator expects exactly one pipeline.");
        }
        return new MultiplyUnit(std::move(pipes[0]));
    }
};

REGISTER_COMPONENT("multiply_unit", MultiplyUnitCreator);

} // namespace PE