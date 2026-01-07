#include "PE/units/MacArrayUnit.h"
#include <cstdint>
#include <memory>
#include <iostream>

namespace PE {

MacArrayUnit::MacArrayUnit(PipelineFactory mult_pipe_factory, PipelineFactory add_pipe_factory, size_t array_size)
    : size(array_size) {
    for (size_t i = 0; i < size; ++i) {
        mac_units.emplace_back(std::make_unique<MacUnit>(mult_pipe_factory(), add_pipe_factory()));
    }
    reset();
}

void MacArrayUnit::reset() {
    for (auto& mac : mac_units) {
        mac->reset();
    }
}


void MacArrayUnit::clock_cycle() {
    cycle_count++;
    for (auto& mac : mac_units) {
        mac->clock_cycle();
    }
}

bool MacArrayUnit::is_active() const {
    for (const auto& mac : mac_units) {
        if (mac->is_active()) {
            return true;
        }
    }
    return false;
}

void MacArrayUnit::load_inputs(const std::vector<std::pair<Number, Number>>& inputs, const std::vector<bool>& valids, const std::vector<bool>& reset_flags) {
    // std::cout << "[DEBUG] MacArrayUnit::load_inputs size=" << inputs.size() << std::endl;
    if (inputs.size() != size || valids.size() != size) {
        std::cerr << "[ERROR] Input size mismatch in MacArrayUnit" << std::endl;
        throw std::invalid_argument("Input size does not match MacArrayUnit size.");
    }

    if (inputs.size() > size) {
        std::cerr << "[ERROR] Too many inputs in MacArrayUnit" << std::endl;
        throw std::invalid_argument("Too many inputs provided to MacArrayUnit.");
    }

    for (size_t i = 0; i < size; ++i) {
        mac_units[i]->load_inputs(
            i >= inputs.size() ? Number() : inputs[i].first,
            i >= inputs.size() ? Number() : inputs[i].second,
            i >= valids.size() ? false : valids[i],
            (reset_flags.empty() || i >= reset_flags.size()) ? false : reset_flags[i]
        );
    }
}

bool MacArrayUnit::has_final_output() const {
    for (const auto& mac : mac_units) {
        if (!mac->has_final_output()) {
            return false;
        }
    }
    return true;
}

bool MacArrayUnit::has_output(size_t mac_index) const {
    if (mac_index >= size) {
        return false; 
    }
    return mac_units[mac_index]->has_final_output();
}

Number MacArrayUnit::pop_output(size_t mac_index) {
    if (mac_index >= size) {
        std::cerr << "[ERROR] Mac index out of range: " << mac_index << std::endl;
        throw std::out_of_range("Mac index out of range in MacArrayUnit.");
    }
    return mac_units[mac_index]->get_final_output();
}

}// namespace PE