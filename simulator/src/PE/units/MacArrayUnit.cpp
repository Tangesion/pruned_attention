#include "PE/units/MacArrayUnit.h"
#include <memory>

namespace PE {

MacArrayUnit::MacArrayUnit(PipelinePtr mult_pipe, PipelinePtr add_pipe, size_t array_size)
    : size(array_size) {
    for (size_t i = 0; i < size; ++i) {
        mac_units.emplace_back(std::make_unique<MacUnit>(std::move(mult_pipe), std::move(add_pipe)));
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

void MacArrayUnit::load_inputs(const std::vector<std::pair<uint16_t, uint16_t>>& inputs, const std::vector<bool>& valids) {
    if (inputs.size() != size || valids.size() != size) {
        throw std::invalid_argument("Input size does not match MacArrayUnit size.");
    }

    if (inputs.size() > size) {
        throw std::invalid_argument("Too many inputs provided to MacArrayUnit.");
    }

    for (size_t i = 0; i < size; ++i) {
        mac_units[i]->load_inputs(
            i >= inputs.size() ? 0 : inputs[i].first,
            i >= inputs.size() ? 0 : inputs[i].second,
            i >= valids.size() ? false : valids[i]
        );
    }
}

std::vector<uint16_t> MacArrayUnit::pop_all_outputs() {
    std::vector<uint16_t> results;
    results.reserve(size);

    for (auto& mac : mac_units) {
        if (mac->has_final_output()) {
            results.push_back(mac->get_final_output());
        } else {
            throw std::runtime_error("MacArrayUnit: One of the MacUnits has no output to pop.");
        }
    }
    return results;
}


}// namespace PE