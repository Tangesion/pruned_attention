#include "PE/scheduler/GemvScheduler.h"
#include "PE/units/ReduceMacArrayUnit.h"
#include "bf16/add_sim.h"
#include "bf16/multiply_sim.h"
#include "bf16/bf16_basic_ops.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <memory>

using namespace PE;

void test_single_batch_gemv() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Testing GemvScheduler Single Batch GEMV" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // 1. Setup Scheduler
    size_t num_pes = 4;
    double bandwidth = 32.0; // Assume sufficient bandwidth
    
    // Create pipelines
    // Note: Pipelines are moved into MacArrayUnit, which is moved into GemvScheduler
    // Since MacArrayUnit takes ownership of pipelines, and GemvScheduler takes ownership of MacArrayUnit
    // We need to create new pipelines for each test or inside the test setup
    
    // Create pipelines factories
    auto mult_pipe_factory = []() { return std::make_unique<bf16::BF16MultiplyPipeline>(); };
    auto add_pipe_factory = []() { return std::make_unique<bf16::BF16AddPipeline>(); };
    
    auto mac_array = std::make_unique<ReduceMacArrayUnit>(mult_pipe_factory, add_pipe_factory, num_pes);

    GemvScheduler::Config config{num_pes};
    GemvScheduler scheduler(std::move(mac_array), config);

    // 2. Prepare Data
    // Matrix: 4x4
    // [ 1, 2, 3, 4 ]
    // [ 5, 6, 7, 8 ]
    // [ 9, 10, 11, 12 ]
    // [ 13, 14, 15, 16 ]
    
    // Input: 1x4
    // [ 1, 0, 1, 0 ]
    
    // Expected Output: 1x4
    // [ 1*1 + 2*0 + 3*1 + 4*0 ] = 4
    // [ 5*1 + 6*0 + 7*1 + 8*0 ] = 12
    // [ 9*1 + 10*0 + 11*1 + 12*0 ] = 20
    // [ 13*1 + 14*0 + 15*1 + 16*0 ] = 28

    int K = 4;
    int N = 4;

    std::vector<std::vector<float>> matrix_f = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {5.0f, 6.0f, 7.0f, 8.0f},
        {9.0f, 10.0f, 11.0f, 12.0f},
        {13.0f, 14.0f, 15.0f, 16.0f}
    };

    std::vector<float> input_f = {1.0f, 0.0f, 1.0f, 0.0f};
    std::vector<float> expected_f = {4.0f, 12.0f, 20.0f, 28.0f};

    // Convert to bf16
    std::vector<std::vector<uint16_t>> matrix_bf16(K, std::vector<uint16_t>(N));
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix_bf16[i][j] = bf16::float_to_bf16(matrix_f[i][j]);
        }
    }

    std::vector<uint16_t> input_bf16(K);
    for (int i = 0; i < K; ++i) {
        input_bf16[i] = bf16::float_to_bf16(input_f[i]);
    }

    // 3. Run GEMV
    // Matrix dimensions in run_gemv doc says [K x N], input [1, K], return [1, N]
    // My matrix_bf16 is [rows][cols] which is 4x4.
    // Usually K is inner dimension, N is output dimension. 
    // M x K * K x N = M x N. Here M=1.
    // So Matrix should be K x N. 
    // Wait, the scheduler doc says:
    // matrix: [K x N]
    // input:  [1, K]
    // return: [1, N]
    // This implies Matrix rows = K, cols = N ??
    // Standard GEMV: y = A * x. 
    // If A is MxN (M rows, N cols), x is Nx1. result y is Mx1.
    // Here we have vector * matrix? 
    // input [1, K] * matrix [K, N] = [1, N]. 
    // So input is 1xK (row vector). Matrix has K rows and N cols.
    // So Matrix[i][j] where i in [0, K-1], j in [0, N-1].
    
    // In my example:
    // Input is 1x4 (K=4). Matrix is 4x4 (K=4, N=4).
    // Result is 1x4 (N=4).
    
    // Let's recheck the logic.
    // [1, 0, 1, 0] * 
    // [ 1, 2, 3, 4 ]
    // [ 5, 6, 7, 8 ]
    // [ 9, 10, 11, 12 ]
    // [ 13, 14, 15, 16 ]
    
    // Row 0 of Matrix corresponds to 1st element of input? 
    // If it is vector-matrix multiplication:
    // res[j] = sum(input[i] * matrix[i][j])
    
    // i=0: input[0]=1. matrix[0][0..3] = 1,2,3,4. 
    // i=1: input[1]=0. matrix[1][0..3] = 5,6,7,8.
    // i=2: input[2]=1. matrix[2][0..3] = 9,10,11,12.
    // i=3: input[3]=0. matrix[3][0..3] = 13,14,15,16.
    
    // Sum:
    // j=0: 1*1 + 0*5 + 1*9 + 0*13 = 10
    // j=1: 1*2 + 0*6 + 1*10 + 0*14 = 12
    // j=2: 1*3 + 0*7 + 1*11 + 0*15 = 14
    // j=3: 1*4 + 0*8 + 1*12 + 0*16 = 16
    
    // So result is [10, 12, 14, 16].
    
    // Wait, my previous manual calculation was:
    // [ 1*1 + 2*0 + 3*1 + 4*0 ] = 4  <-- This assumed Matrix * vector (standard gemv) where vector is column. 
    // If Matrix * vector:
    // Row 0: [1,2,3,4] . [1,0,1,0] = 1*1 + 2*0 + 3*1 + 4*0 = 4.
    // Row 1: [5,6,7,8] . [1,0,1,0] = 5*1 + 6*0 + 7*1 + 8*0 = 12.
    // ...
    
    // But the signature says: run_gemv(matrix, input).
    // matrix: [K x N]
    // input:  [1, K]
    // return: [1, N]
    
    // This signature [1, K] * [K, N] = [1, N] is vector-matrix multiplication. x * A.
    // So my second manual calculation (resulting in 10, 12, 14, 16) is correct for x * A.
    
    // However, usually "Gemv" refers to Matrix-Vector multiplication. 
    // If it's Matrix-Vector, then input should be [N] or [K] depending on definition.
    // If Matrix is M x N, vector should be N x 1. Result M x 1.
    
    // Let's stick to the signature documentation in GemvScheduler.h:
    // matrix: [K x N]
    // input:  [1, K]
    // return: [1, N]
    
    // So it is x * A.
    // Expected output: [10.0f, 12.0f, 14.0f, 16.0f]
    
    std::vector<float> expected_xA = {10.0f, 12.0f, 14.0f, 16.0f};

    std::vector<uint16_t> result_bf16 = scheduler.run_gemv(matrix_bf16, input_bf16);

    // 4. Verify
    assert(result_bf16.size() == N);
    
    std::cout << "Result: ";
    for (size_t i = 0; i < N; ++i) {
        float val = bf16::bf16_to_float(result_bf16[i]);
        std::cout << val << " ";
        float err = std::abs(val - expected_xA[i]);
        if (err > 0.5f) { // bf16 precision might be low, allow 0.5 error for now
             std::cerr << "Mismatch at index " << i << ": expected " << expected_xA[i] << ", got " << val << std::endl;
        }
        assert(err < 0.5f);
    }
    std::cout << std::endl;

    std::cout << "Single Batch GEMV Passed." << std::endl;
}

int main() {
    test_single_batch_gemv();
    return 0;
}
