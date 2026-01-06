#pragma once

#include "PE/base/PipelineInput.h"
#include <cstdint>
#include <deque>
#include <memory>

namespace PE {

template <typename T>
class Pipeline {
public:
    virtual ~Pipeline() = default;

    virtual void clock_cycle(const PipelineInput& input) = 0;
    virtual void reset() = 0;
    virtual const std::deque<T>& get_outputs() const = 0;
    virtual T pop_output() = 0;
    virtual bool is_active() const = 0;

protected:
    std::deque<T> outputs;
    uint32_t cycle_count;
};

template <typename T>
using PipelinePtr = std::unique_ptr<Pipeline<T>>;

} // namespace PE
