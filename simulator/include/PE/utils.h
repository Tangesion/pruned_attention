#ifndef UTILS_H
#define UTILS_H
#include <memory>
#include "bf16/add_sim.h"
#include "bf16/multiply_sim.h"
#include <cstdint>

namespace pe {

class MultiplyUnit {
public:
    MultiplyUnit();
    ~MultiplyUnit();

    bool is_input_valid() const;
    bool is_output_valid() const;
    void set_inputs(uint16_t a, uint16_t b, bool valid);
    void clock_cycle();
    void reset();


private:
    uint16_t bf16_a;
    uint16_t bf16_b;
    uint16_t output;
    bool input_valid;
    bool output_valid;
    std::unique_ptr<bf16::BF16MultiplyPipeline> pipeline;

};

class AddUnit {
public:
    AddUnit();
    ~AddUnit();

    bool is_input_valid() const;
    bool is_output_valid() const;
    void set_inputs(uint16_t a, uint16_t b, bool valid);
    void clock_cycle();
    void reset();

private:
    uint16_t bf16_a;
    uint16_t bf16_b;
    uint16_t output;
    bool input_valid;
    bool output_valid;
    std::unique_ptr<bf16::BF16AddPipeline> pipeline;

};

class MacUnit {
public:
    MacUnit();
    ~MacUnit();

    bool is_input_valid() const;
    bool is_output_valid() const;
    void set_inputs(uint16_t a, uint16_t b, bool valid);
    void clock_cycle();

private:
    uint16_t bf16_a;
    uint16_t bf16_b;

    


};

} // namespace pe

#endif // UTILS_H