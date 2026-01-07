#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <stack>
#include <vector>

#include <functional>
#include "PE/base/ComputeComponent.h"
#include "PE/base/Pipeline.h"
#include "PE/base/DataType.h"
#include "PE/units/AddUnit.h"


namespace PE {

class AddTreeUnit : public ComputeComponent {

public:
    using PipelineFactory = std::function<PipelinePtr<Number>()>;
    AddTreeUnit(PipelineFactory add_pipe_factory, size_t input_num);
    void reset() override;
    bool is_active() const override;
    void clock_cycle() override;
    
    // Check if the final result is ready
    bool has_output() const;
    // Get the final result
    Number get_output();
    void load_inputs(const std::vector<Number>& inputs, const bool valid);

private:

    size_t compute_tree_height(size_t unit_num) const;

    std::vector<std::vector<std::unique_ptr<AddUnit>>> tree;

    std::vector<bool> stage_valids;
    std::vector<std::deque<Number>> stage_outputs;

    std::deque<Number> outputs;
    
    size_t input_num;
    size_t tree_height;
    bool input_valid = false;

    size_t cycle_count = 0;

};

} // namespace PE