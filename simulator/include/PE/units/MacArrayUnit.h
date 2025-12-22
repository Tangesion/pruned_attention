#pragma once
#include <vector>
#include "PE/base/ComputeComponent.h"
#include "PE/base/Pipeline.h"
#include "PE/units/MacUnit.h"

namespace PE {

class MacArrayUnit: public ComputeComponent {
public:
    MacArrayUnit(PipelinePtr mult_pipe, PipelinePtr add_pipe, size_t array_size);
    void reset() override;
    bool is_active() const override;
    void clock_cycle() override;

    void load_inputs(const std::vector<std::pair<uint16_t, uint16_t>>& inputs, const std::vector<bool>& valids); 
    std::vector<uint16_t> pop_all_outputs();

private:
    std::vector<std::unique_ptr<MacUnit>> mac_units;
    size_t size;
    uint32_t cycle_count = 0;
};

}// namespace PE