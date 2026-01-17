#include "PE/scheduler/GemvScheduler.h"
#include "int4/multiply_sim.h"
#include "int4/add_sim.h"
#include <iostream>
#include <vector>
#include <cassert>

using namespace PE;

void test_int4_gemv_basic() {
    std::cout << "Testing Int4 GEMV Basic (All 1s)..." << std::endl;

    // 1. Hardware Setup
    auto mult_factory = []() { return std::make_unique<int4::Int4MultiplyPipeline>(1); };
    auto add_factory = []() { return std::make_unique<int4::Int4AddPipeline>(1); };
    
    auto mac_array = std::make_unique<ReduceMacArrayUnit>(mult_factory, add_factory, 4);
    GemvScheduler::Config config;
    config.num_pes = 4;
    
    GemvScheduler scheduler(std::move(mac_array), config);

    // 2. Data Setup (4x4 matrix of 1s, 1x4 vector of 1s)
    // Int4 '1' is just 0x1 in bits
    std::vector<std::vector<uint16_t>> matrix(4, std::vector<uint16_t>(4, 1));
    std::vector<uint16_t> input(4, 1);

    // 3. Run
    std::vector<int32_t> result = scheduler.run_gemv_int32(matrix, input);

    // 4. Verify
    // Each element should be 1*1 + 1*1 + 1*1 + 1*1 = 4
    std::cout << "Result: ";
    for (auto v : result) std::cout << v << " ";
    std::cout << std::endl;

    assert(result.size() == 4);
    for (auto v : result) assert(v == 4);
    std::cout << "PASSED." << std::endl;
}

void test_int4_gemv_signed() {
    std::cout << "Testing Int4 GEMV Signed..." << std::endl;

    auto mult_factory = []() { return std::make_unique<int4::Int4MultiplyPipeline>(1); };
    auto add_factory = []() { return std::make_unique<int4::Int4AddPipeline>(1); };
    
    auto mac_array = std::make_unique<ReduceMacArrayUnit>(mult_factory, add_factory, 4);
    GemvScheduler::Config config;
    config.num_pes = 4;
    
    GemvScheduler scheduler(std::move(mac_array), config);

    // Matrix:
    // [ -1,  2 ]
    // [  3, -4 ]
    // In 4-bit signed: -1 = 0xF, -4 = 0xC
    std::vector<std::vector<uint16_t>> matrix = {
        { 0xF, 2 },
        { 3, 0xC }
    };
    // Vector: [ 2, 3 ]
    std::vector<uint16_t> input = { 2, 3 };

    // Result:
    // Row 0: (-1 * 2) + (2 * 3) = -2 + 6 = 4
    // Row 1: (3 * 2) + (-4 * 3) = 6 - 12 = -6
    // Wait, matrix indexing in run_gemv is matrix[k][n] where k is input dim, n is output dim.
    // Matrix[2][2]
    // k=0: [-1, 2]
    // k=1: [3, -4]
    // input = [2, 3] (size K=2)
    // output[0] = matrix[0][0]*input[0] + matrix[1][0]*input[1] = -1*2 + 3*3 = -2 + 9 = 7
    // output[1] = matrix[0][1]*input[0] + matrix[1][1]*input[1] = 2*2 + -4*3 = 4 - 12 = -8

    std::vector<int32_t> result = scheduler.run_gemv_int32(matrix, input);

    std::cout << "Result: ";
    for (auto v : result) std::cout << v << " ";
    std::cout << std::endl;

    assert(result.size() == 2);
    assert(result[0] == 7);
    assert(result[1] == -8);
    std::cout << "PASSED." << std::endl;
}

int main() {
    test_int4_gemv_basic();
    test_int4_gemv_signed();
    return 0;
}
