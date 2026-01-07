#include "int4/multiply_sim.h"
#include "int4/add_sim.h"
#include <iostream>
#include <cassert>
#include <memory>

void test_multiply() {
    std::cout << "Testing Int4MultiplyPipeline..." << std::endl;
    int4::Int4MultiplyPipeline pipe(2); // Latency 2

    // Case 1: 2 * 3 = 6
    PE::TwoOperandInput in1;
    in1.a = PE::Number(2);
    in1.b = PE::Number(3);
    in1.valid = true;

    pipe.clock_cycle(in1); // Cycle 1: Input fed. Ready at 1+2=3.
    assert(pipe.get_outputs().empty()); 

    PE::TwoOperandInput bubble;
    bubble.valid = false;
    pipe.clock_cycle(bubble); // Cycle 2
    assert(pipe.get_outputs().empty());

    pipe.clock_cycle(bubble); // Cycle 3: Output should be ready 
    
    assert(!pipe.get_outputs().empty());
    int32_t res = pipe.pop_output().as_int32();
    assert(res == 6);
    std::cout << "  2 * 3 = " << res << " [PASS]" << std::endl;

    // Case 2: -2 * 3 = -6
    // -2 in 4-bit 2's complement is 1110 (14 or 0xE)
    in1.a = PE::Number(0xE);
    in1.b = PE::Number(3);
    in1.valid = true;
    pipe.clock_cycle(in1); // Cycle 4. Ready at 4+2=6.
    
    pipe.clock_cycle(bubble); // Cycle 5.
    assert(pipe.get_outputs().empty());
    
    pipe.clock_cycle(bubble); // Cycle 6.
    assert(!pipe.get_outputs().empty());
    res = pipe.pop_output().as_int32();
    assert(res == -6);
    std::cout << "  -2 * 3 = " << res << " [PASS]" << std::endl;
    
    std::cout << "Int4MultiplyPipeline Passed." << std::endl;
}

void test_add() {
    std::cout << "Testing Int4AddPipeline (Int32 addition)..." << std::endl;
    int4::Int4AddPipeline pipe(1); // Latency 1

    // Case 1: 10 + 20 = 30
    PE::TwoOperandInput in1;
    in1.a = PE::Number(10);
    in1.b = PE::Number(20);
    in1.valid = true;

    pipe.clock_cycle(in1); // Cycle 1. Ready at 2.
    assert(pipe.get_outputs().empty());

    PE::TwoOperandInput bubble; 
    bubble.valid = false;
    
    pipe.clock_cycle(bubble); // Cycle 2. Ready.
    assert(!pipe.get_outputs().empty());
    int32_t res = pipe.pop_output().as_int32();
    assert(res == 30);
    std::cout << "  10 + 20 = " << res << " [PASS]" << std::endl;

    std::cout << "Int4AddPipeline Passed." << std::endl;
}

int main() {
    test_multiply();
    test_add();
    return 0;
}