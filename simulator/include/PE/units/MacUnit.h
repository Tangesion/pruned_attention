#pragma once

#include "PE/base/ComputeComponent.h"
#include "PE/units/AddUnit.h"
#include "PE/units/MultiplyUnit.h"
#include <deque>
#include <memory>

namespace PE {

class MacUnit : public ComputeComponent {
public:
    MacUnit(std::shared_ptr<MultiplyUnit> mult, std::shared_ptr<AddUnit> add);

    void reset() override;
    bool is_active() const override;
    void clock_cycle() override;

    void load_inputs(uint16_t in1, uint16_t in2, bool valid);
    void set_initial_acc(uint16_t initial_acc);
    bool has_final_output() const;
    uint16_t get_final_output();
    uint16_t get_accumulated_result() const;

private:
    std::shared_ptr<MultiplyUnit> mult_unit;
    std::shared_ptr<AddUnit> add_unit;

    uint16_t input1 = 0;
    uint16_t input2 = 0;
    bool input_valid = false;

    std::deque<uint16_t> multiply_queue;
    std::deque<uint16_t> acc_queue;

    std::deque<uint16_t> outputs;
    
    uint32_t cycle_count = 0;
};

} // namespace PE
