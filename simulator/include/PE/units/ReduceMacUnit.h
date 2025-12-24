#pragma once
#include "PE/base/ComputeComponent.h"
#include "PE/units/MacArrayUnit.h"
#include "PE/units/AddTreeUnit.h"
#include <cstdint>
#include <deque>
#include <vector>
#include <memory>
#include <functional>

namespace PE {

class ReduceMacUnit : public ComputeComponent {
public:
    using PipelineFactory = std::function<PipelinePtr()>;
    
    ReduceMacUnit(PipelineFactory mult_pipe_factory, PipelineFactory add_pipe_factory);
    
    void reset() override;
    bool is_active() const override;
    void clock_cycle() override;

    // Load inputs for the MacArray
    void load_inputs(uint16_t in1, uint16_t in2, bool valid, bool reset_flag=false);

    // Check and get final reduced result
    bool has_output() const;
    uint16_t get_output();

private:
    std::unique_ptr<MacUnit> mac_unit;
    std::unique_ptr<AddTreeUnit> add_tree;
    size_t cycle_count = 0;

    std::deque<uint16_t> mac_outputs;
};

} // namespace PE
