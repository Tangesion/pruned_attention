#include "PE/units/ReduceMacUnit.h"
#include "bf16/add_sim.h"
#include <stdexcept>
#include <vector>

namespace PE {

ReduceMacUnit::ReduceMacUnit(PipelineFactory mult_pipe_factory, PipelineFactory add_pipe_factory) {
    mac_unit = std::make_unique<MacUnit>(mult_pipe_factory(), add_pipe_factory());
    add_tree = std::make_unique<AddTreeUnit>(add_pipe_factory, ADD_PIPELINE_DEPTH);
}

void ReduceMacUnit::reset() {
    mac_unit->reset();
    add_tree->reset();
    mac_outputs.clear();
    cycle_count = 0;
}

bool ReduceMacUnit::is_active() const {
    return mac_unit->is_active() || add_tree->is_active();
}

void ReduceMacUnit::load_inputs(Number in1,
                                Number in2,
                                bool valid,
                                bool reset_flag) { 
    
    mac_unit->load_inputs(in1, in2, valid, reset_flag);
}

bool ReduceMacUnit::has_output() const {
    return add_tree->has_output();
}

Number ReduceMacUnit::get_output() {
    return add_tree->get_output();
}

void ReduceMacUnit::clock_cycle() {
    cycle_count++;
    bool add_tree_input_ready = mac_outputs.size() == ADD_PIPELINE_DEPTH;

    if (add_tree_input_ready) {
        std::vector<Number> inputs;
        for (size_t i = 0; i < ADD_PIPELINE_DEPTH; ++i) {
            inputs.push_back(mac_outputs.front());
            mac_outputs.pop_front();
        }
        add_tree->load_inputs(inputs, true);
    }
    else {
        // Need to provide inputs of type Number. Default Number() is 0.
        add_tree->load_inputs(std::vector<Number>(ADD_PIPELINE_DEPTH, Number()), false);
    }

    add_tree->clock_cycle();

    mac_unit->clock_cycle();
    if (mac_unit->has_final_output()) {
        Number mac_out = mac_unit->get_final_output();
        mac_outputs.push_back(mac_out);
    }
    
}

} // namespace PE