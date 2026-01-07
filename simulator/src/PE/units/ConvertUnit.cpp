#include "PE/units/ConvertUnit.h"
#include "PE/base/backend.h"
#include <stdexcept>
#include <vector>

namespace PE {

ConvertUnit::ConvertUnit(PipelinePtr<Number> p) : pipeline(std::move(p)) {
    if (!pipeline) {
        throw std::invalid_argument("ConvertUnit received a null pipeline.");
    }
}

void ConvertUnit::reset() {
    pipeline->reset();
    current_input = {};
}

bool ConvertUnit::is_active() const {
    return pipeline->is_active();
}

void ConvertUnit::clock_cycle() {
    pipeline->clock_cycle(current_input);
    current_input.valid = false;
}

void ConvertUnit::load_operand(uint32_t val, bool valid) {
    current_input.val = val;
    current_input.valid = valid;
}

Number ConvertUnit::get_result() {
    return pipeline->pop_output();
}

bool ConvertUnit::has_output() const {
    return !pipeline->get_outputs().empty();
}

class ConvertUnitCreator : public Backend::Creator {
public:
    ComputeComponent* onCreate(std::vector<PipelinePtr<Number>>&& pipes) const override {
        if (pipes.size() != 1) {
            throw std::invalid_argument("ConvertUnitCreator expects exactly one pipeline.");
        }
        return new ConvertUnit(std::move(pipes[0]));
        
    }
};

REGISTER_COMPONENT("convert_unit", ConvertUnitCreator);

} // namespace PE