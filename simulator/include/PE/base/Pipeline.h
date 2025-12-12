#pragma once

#include "PE/base/PipelineInput.h"
#include <cstdint>
#include <deque>
#include <memory>

namespace PE {

class Pipeline {
public:
    virtual ~Pipeline() = default;

    virtual void clock_cycle(const PipelineInput& input) = 0;
    virtual void reset() = 0;
    virtual const std::deque<uint16_t>& get_outputs() const = 0;
    virtual uint16_t pop_output() = 0;
    virtual bool is_active() const = 0;

protected:
    std::deque<uint16_t> outputs;
    uint32_t cycle_count;
};

using PipelinePtr = std::unique_ptr<Pipeline>;

} // namespace PE
