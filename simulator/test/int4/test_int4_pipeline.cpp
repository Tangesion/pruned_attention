#include "int4/int4_pipeline.h"
#include <iostream>
#include <cassert>
#include <memory>

void test_multiply() {
    std::cout << "Testing Int4MultiplyPipeline..." << std::endl;
    int4::Int4MultiplyPipeline pipe(2); // Latency 2

    // Case 1: 2 * 3 = 6
    PE::TwoOperandInput in1;
    in1.a = 2;
    in1.b = 3;
    in1.valid = true;

    pipe.clock_cycle(in1); // Cycle 1: Input fed. Ready at 1+2=3.
    assert(pipe.get_outputs().empty()); 

    PE::TwoOperandInput bubble;
    bubble.valid = false;
    pipe.clock_cycle(bubble); // Cycle 2
    assert(pipe.get_outputs().empty());

    pipe.clock_cycle(bubble); // Cycle 3: Output should be ready? 
    
    assert(!pipe.get_outputs().empty());
    int32_t res = pipe.pop_output();
    assert(res == 6);
    std::cout << "  2 * 3 = " << res << " [PASS]" << std::endl;

    // Case 2: -2 * 3 = -6
    // -2 in 4-bit 2's complement is 1110 (14 or 0xE)
    in1.a = 0xE;
    in1.b = 3;
    in1.valid = true;
    pipe.clock_cycle(in1); // Cycle 4. Ready at 4+2=6.
    
    pipe.clock_cycle(bubble); // Cycle 5.
    assert(pipe.get_outputs().empty());
    
    pipe.clock_cycle(bubble); // Cycle 6.
    assert(!pipe.get_outputs().empty());
    res = pipe.pop_output();
    assert(res == -6);
    std::cout << "  -2 * 3 = " << res << " [PASS]" << std::endl;
    
    std::cout << "Int4MultiplyPipeline Passed." << std::endl;
}

void test_mac() {
    std::cout << "Testing Int4MacPipeline..." << std::endl;
    int4::Int4MacPipeline pipe(1); // Latency 1

    // Case 1: 2 * 3 + 10 = 16
    int4::Int4MacInput in1;
    in1.a = 2;
    in1.b = 3;
    in1.c = 10;
    in1.valid = true;

    pipe.clock_cycle(in1); // Cycle 1. Ready at 1+1=2.
    assert(pipe.get_outputs().empty());

    PE::TwoOperandInput bubble; 
    bubble.valid = false;
    
    pipe.clock_cycle(bubble); // Cycle 2. Ready.
    assert(!pipe.get_outputs().empty());
    int32_t res = pipe.pop_output();
    assert(res == 16);
    std::cout << "  2 * 3 + 10 = " << res << " [PASS]" << std::endl;

    // Case 2: -2 * 3 + 10 = 4
    in1.a = 0xE; // -2
    in1.b = 3;
    in1.c = 10;
    in1.valid = true;
    
    pipe.clock_cycle(in1); // Cycle 3. Ready at 4.
    pipe.clock_cycle(bubble); // Cycle 4.
    
    assert(!pipe.get_outputs().empty());
    res = pipe.pop_output();
    assert(res == 4);
    std::cout << "  -2 * 3 + 10 = " << res << " [PASS]" << std::endl;

    std::cout << "Int4MacPipeline Passed." << std::endl;
}

int main() {
    test_multiply();
    test_mac();
    return 0;
}
