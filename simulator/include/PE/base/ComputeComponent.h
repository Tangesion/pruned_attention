#pragma once

#include <memory>

namespace PE {

class ComputeComponent {
public:
    virtual ~ComputeComponent() = default;

    // A parameter-less clock cycle representing a single 'tick' of the component.
    virtual void clock_cycle() = 0;

    // Resets the component to its initial state.
    virtual void reset() = 0;

    // Checks if the component is currently processing data.
    virtual bool is_active() const = 0;
};

// Define a shared pointer type for components for easy composition.
using ComponentPtr = std::shared_ptr<ComputeComponent>;

} // namespace PE
