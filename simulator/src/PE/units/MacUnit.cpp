#include "PE/units/MacUnit.h"
#include "PE/base/Pipeline.h"
#include "PE/base/backend.h"
#include "bf16/add_sim.h"
#include <stdexcept>
#include <memory>
#include <vector>

namespace PE {

MacUnit::MacUnit(PipelinePtr<uint16_t> mult_pipe, PipelinePtr<uint16_t> add_pipe)
    : mult_unit(std::make_unique<MultiplyUnit>(std::move(mult_pipe))),
      add_unit(std::make_unique<AddUnit>(std::move(add_pipe))) {
    if (!mult_unit || !add_unit) {
        throw std::invalid_argument("MacUnit components cannot be null.");
    }
    reset();
}

void MacUnit::reset() {
    mult_unit->reset();
    add_unit->reset();
    
    input1 = 0;
    input2 = 0;
    input_valid = false;
    
    multiply_queue.clear();
    reset_queue.clear();
    outputs.clear();

    // Initialize accumulator queue with zeros to prevent bubbles
    acc_queue.clear();
    for (size_t i = 0; i < ADD_PIPELINE_DEPTH; ++i) {
        acc_queue.push_back(0);
    }

    cycle_count = 0;
}

bool MacUnit::is_active() const {
    return mult_unit->is_active() || add_unit->is_active();
}

void MacUnit::load_inputs(uint16_t in1, uint16_t in2, bool valid, bool reset_flag) {
    this->input1 = in1;
    this->input2 = in2;
    this->input_valid = valid;
    this->reset_flag = reset_flag;
}

void MacUnit::set_initial_acc(uint16_t initial_acc) {
    acc_queue.clear();
    for (size_t i = 0; i < ADD_PIPELINE_DEPTH; ++i)
        acc_queue.push_back(initial_acc);
}

bool MacUnit::has_final_output() const {
    return !outputs.empty();
}

uint16_t MacUnit::get_final_output() {
    if (outputs.empty()) {
        throw std::runtime_error("No final output available from MacUnit.");
    }
    uint16_t val = outputs.front();
    outputs.pop_front();
    return val;
}

uint16_t MacUnit::get_accumulated_result() const {
    if (acc_queue.empty()) {
        return 0;
    }
    return acc_queue.front();
}

void MacUnit::clock_cycle() {
    cycle_count++;

    bool add_inputs_valid = !acc_queue.empty() && !multiply_queue.empty();
    
    if (add_inputs_valid) {
        uint16_t mult_val = multiply_queue.front();
        bool reset_acc = reset_queue.front();
        uint16_t acc_val = 0;

        if (!reset_acc) {
            acc_val = acc_queue.front();
            acc_queue.pop_front();
        } else {
            outputs.push_back(acc_queue.front());
            acc_queue.pop_front();
            acc_val = 0;
        }
        reset_queue.pop_front();
        multiply_queue.pop_front();
        add_unit->load_operands(mult_val, acc_val, true);
    } else {
        add_unit->load_operands(0, 0, false);
    }

    add_unit->clock_cycle();
    if (add_unit->has_output()) {
        uint16_t add_result = add_unit->get_result();
        acc_queue.push_back(add_result);
        //outputs.push_back(add_result);
    }

    mult_unit->load_operands(this->input1, this->input2, this->input_valid);
    if (this->input_valid) {
        reset_queue.push_back(this->reset_flag);
    }
    this->reset_flag = false;
    this->input_valid = false; 
    mult_unit->clock_cycle();
    if (mult_unit->has_output()) {
        multiply_queue.push_back(mult_unit->get_result());
        this->reset_flag = false;
    }
}

class MacUnitCreator : public Backend::Creator {
public:
    ComputeComponent* onCreate(std::vector<PipelinePtr<uint16_t>>&& pipes) const override {
        if (pipes.size() != 2) {
            throw std::invalid_argument("MacUnitCreator expects exactly two pipelines.");
        }
        return new MacUnit(std::move(pipes[0]), std::move(pipes[1]));
    }
};

REGISTER_COMPONENT("mac_unit", MacUnitCreator);

} // namespace PE
