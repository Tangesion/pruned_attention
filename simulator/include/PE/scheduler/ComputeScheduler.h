#pragma once
#include <memory>
#include <cstddef>
#include "PE/base/ComputeComponent.h"


namespace PE {

class ComputeScheduler {
public:
    ComputeScheduler(std::unique_ptr<ComputeComponent> component)
        : hardware_component(std::move(component)), cycle_count(0) {};

    void reset() {
        cycle_count = 0;
        hardware_component->reset();
    };

    size_t get_total_cycles() const {
        return cycle_count;
    };

protected:
    void clock_cycle() {
        cycle_count++;
        hardware_component->clock_cycle();
    }

protected:
    std::unique_ptr<ComputeComponent> hardware_component;
    size_t cycle_count = 0;

};

} // namespace PE