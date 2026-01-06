#pragma once
#include <cstdint>
#include <sys/types.h>
#include <vector>
#include <functional>
#include "PE/base/ComputeComponent.h"
#include "PE/base/Pipeline.h"
#include "PE/units/MacUnit.h"

namespace PE {

class MacArrayUnit: public ComputeComponent {
public:
    using PipelineFactory = std::function<PipelinePtr<uint16_t>()>;
    MacArrayUnit(PipelineFactory mult_pipe_factory, PipelineFactory add_pipe_factory, size_t array_size);
    void reset() override;
    bool is_active() const override;
    void clock_cycle() override;

    void load_inputs(const std::vector<std::pair<uint16_t, uint16_t>>& inputs, const std::vector<bool>& valids, const std::vector<bool>& reset_flags = {}); 
    uint16_t pop_output(size_t mac_index);
    bool has_final_output() const;
    bool has_output(size_t mac_index) const;

private:
    std::vector<std::unique_ptr<MacUnit>> mac_units;
    size_t size;
    uint32_t cycle_count = 0;
};

}// namespace PE