#include "PE/units/ReduceMacArrayUnit.h"
#include <iostream>
#include <stdexcept>

namespace PE {

ReduceMacArrayUnit::ReduceMacArrayUnit(PipelineFactory mult_pipe_factory, PipelineFactory add_pipe_factory, size_t array_size)
    : size(array_size) {
    for (size_t i = 0; i < size; ++i) {
        // ReduceMacUnit constructor takes (mult_pipe_factory, add_pipe_factory)
        reduce_mac_units.emplace_back(std::make_unique<ReduceMacUnit>(mult_pipe_factory, add_pipe_factory));
    }
    reset();
}

void ReduceMacArrayUnit::reset() {
    for (auto& unit : reduce_mac_units) {
        unit->reset();
    }
    cycle_count = 0;
}

void ReduceMacArrayUnit::clock_cycle() {
    cycle_count++;
    for (auto& unit : reduce_mac_units) {
        unit->clock_cycle();
    }
}

bool ReduceMacArrayUnit::is_active() const {
    for (const auto& unit : reduce_mac_units) {
        if (unit->is_active()) {
            return true;
        }
    }
    return false;
}

void ReduceMacArrayUnit::load_inputs(const std::vector<std::pair<uint16_t, uint16_t>>& inputs, const std::vector<bool>& valids, const std::vector<bool>& reset_flags) {
    if (inputs.size() != size || valids.size() != size) {
        std::cerr << "[ERROR] Input size mismatch in ReduceMacArrayUnit" << std::endl;
        throw std::invalid_argument("Input size does not match ReduceMacArrayUnit size.");
    }
    
    for (size_t i = 0; i < size; ++i) {
        bool reset = (reset_flags.empty() || i >= reset_flags.size()) ? false : reset_flags[i];
        reduce_mac_units[i]->load_inputs(inputs[i].first, inputs[i].second, valids[i], reset);
    }
}

bool ReduceMacArrayUnit::has_output(size_t index) const {
    if (index >= size) {
        return false;
    }
    return reduce_mac_units[index]->has_output();
}

uint16_t ReduceMacArrayUnit::pop_output(size_t index) {
    if (index >= size) {
        std::cerr << "[ERROR] Index out of range: " << index << std::endl;
        throw std::out_of_range("Index out of range in ReduceMacArrayUnit.");
    }
    // ReduceMacUnit uses get_output(), mimicking MacArrayUnit's pop_output name.
    return reduce_mac_units[index]->get_output();
}

} // namespace PE
