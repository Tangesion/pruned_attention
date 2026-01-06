#include "PE/units/AddUnit.h"
#include "PE/base/Pipeline.h"
#include "PE/base/backend.h"
#include <stdexcept>
#include <vector>

namespace PE {

AddUnit::AddUnit(PipelinePtr<uint16_t> p) : pipeline(std::move(p)) {
    if (!pipeline) {
        throw std::invalid_argument("AddUnit received a null pipeline.");
    }
}

void AddUnit::reset() {
    pipeline->reset();
    current_input = {};
}

bool AddUnit::is_active() const {
    return pipeline->is_active();
}

void AddUnit::clock_cycle() {
    pipeline->clock_cycle(current_input);
    current_input.valid = false;
}

void AddUnit::load_operands(uint16_t a, uint16_t b, bool valid) {
    current_input.a = a;
    current_input.b = b;
    current_input.valid = valid;
}

uint16_t AddUnit::get_result() {
    return pipeline->pop_output();
}

bool AddUnit::has_output() const {
    return !pipeline->get_outputs().empty();
}


class AddUnitCreator : public Backend::Creator {
public:
    ComputeComponent* onCreate(std::vector<PipelinePtr<uint16_t>> &&pipes) const override {
        if (pipes.size() != 1) {
            throw std::invalid_argument("AddUnitCreator expects exactly one pipeline.");
        }
        return new AddUnit(std::move(pipes[0]));
    }
};   

REGISTER_COMPONENT("add_unit", AddUnitCreator);

} // namespace PE
