#include "PE/scheduler/GemvScheduler.h"
#include "PE/units/MacArrayUnit.h"
#include "bf16/add_sim.h"
#include "bf16/multiply_sim.h"
#include "bf16/bf16_basic_ops.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <memory>

using namespace PE;

// Helper to generate random float vector
std::vector<float> generate_random_vector(size_t size, float min_val = -0.5f, float max_val = 0.5f) {
    std::vector<float> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (size_t i = 0; i < size; ++i) {
        vec[i] = dis(gen);
    }
    return vec;
}

// Reference GEMV implementation (float)
// Matrix: K x N (row-major: matrix[k][n])
// Input: 1 x K
// Output: 1 x N
std::vector<float> reference_gemv(const std::vector<std::vector<float>>& matrix, const std::vector<float>& input) {
    if (matrix.empty() || input.empty()) return {};
    size_t K = input.size();
    size_t N = matrix[0].size();
    std::vector<float> output(N, 0.0f);

    for (size_t n = 0; n < N; ++n) {
        for (size_t k = 0; k < K; ++k) {
            output[n] += input[k] * matrix[k][n];
        }
    }
    return output;
}

void run_test(size_t K, size_t N, size_t num_pes) {
    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "Test Configuration: Input(K)=" << K << ", Output(N)=" << N << ", Num PEs=" << num_pes << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    // 1. Generate Data
    auto input_f = generate_random_vector(K);
    std::vector<std::vector<float>> matrix_f(K);
    for (size_t k = 0; k < K; ++k) {
        matrix_f[k] = generate_random_vector(N);
    }

    // 2. Compute Reference
    auto expected_f = reference_gemv(matrix_f, input_f);

    // 3. Prepare Simulator Data (BF16)
    std::vector<uint16_t> input_bf16(K);
    for (size_t k = 0; k < K; ++k) {
        input_bf16[k] = bf16::float_to_bf16(input_f[k]);
    }

    std::vector<std::vector<uint16_t>> matrix_bf16(K, std::vector<uint16_t>(N));
    for (size_t k = 0; k < K; ++k) {
        for (size_t n = 0; n < N; ++n) {
            matrix_bf16[k][n] = bf16::float_to_bf16(matrix_f[k][n]);
        }
    }

    // 4. Setup Scheduler
    auto mult_pipe_factory = []() { return std::make_unique<bf16::BF16MultiplyPipeline>(); };
    auto add_pipe_factory = []() { return std::make_unique<bf16::BF16AddPipeline>(); };
    auto mac_array = std::make_unique<MacArrayUnit>(mult_pipe_factory, add_pipe_factory, num_pes);

    GemvScheduler::Config config{32.0, num_pes};
    GemvScheduler scheduler(std::move(mac_array), config);

    // 5. Run Simulation
    std::cout << "Running simulation..." << std::endl;
    auto result_bf16 = scheduler.run_gemv(matrix_bf16, input_bf16);
    
    size_t total_cycles = scheduler.get_total_cycles();
    size_t total_ops = K * N; // MAC operations
    double utilization = (double(total_ops) / (total_cycles * num_pes)) * 100.0;

    std::cout << "Performance Stats:" << std::endl;
    std::cout << "  Total Cycles: " << total_cycles << std::endl;
    std::cout << "  Total MAC Ops: " << total_ops << std::endl;
    std::cout << "  Theoretical Min Cycles (Ideal): " << std::ceil(double(total_ops) / num_pes) << std::endl;
    std::cout << "  Utilization: " << std::fixed << std::setprecision(2) << utilization << "%" << std::endl;

    // 6. Verify Results
    if (result_bf16.size() != N) {
        std::cerr << "Error: Output size mismatch. Expected " << N << ", got " << result_bf16.size() << std::endl;
        return;
    }

    float max_error = 0.0f;
    float total_error = 0.0f;
    float max_rel_error = 0.0f;

    for (size_t i = 0; i < N; ++i) {
        float val = bf16::bf16_to_float(result_bf16[i]);
        float expected = expected_f[i];
        float error = std::abs(val - expected);
        total_error += error;
        max_error = std::max(max_error, error);
        
        if (std::abs(expected) > 1e-5) {
             max_rel_error = std::max(max_rel_error, error / std::abs(expected));
        }
    }

    std::cout << "Verification Result:" << std::endl;
    std::cout << "  Max Absolute Error: " << max_error << std::endl;
    std::cout << "  Avg Absolute Error: " << total_error / N << std::endl;
    
    // BF16 precision is limited (7-8 mantissa bits). 
    // Accumulation error grows with sqrt(K).
    // We check if the result is reasonably close.
    bool passed = true;
    if (max_error > 1.0f && max_rel_error > 0.15f) { // Allow some error
         std::cout << "  Status: WARNING / FAIL (High Error)" << std::endl;
         passed = false;
    } else {
         std::cout << "  Status: PASSED" << std::endl;
    }
    
    // Print a few samples if failed
    if (!passed) {
        std::cout << "Samples (Index: Sim vs Ref):" << std::endl;
        for(size_t i=0; i<std::min(N, (size_t)5); ++i) {
            std::cout << "  " << i << ": " << bf16::bf16_to_float(result_bf16[i]) 
                      << " vs " << expected_f[i] << std::endl;
        }
    }
}

int main() {
    // 1. 标准情况：N 是 NumPEs 的倍数
    // Matrix 4x4, PE=4
    run_test(4, 4, 4);

    // 2. 需要多批次处理 (Batching)：N > NumPEs
    // Matrix 4x8, PE=4 (需要两批)
    run_test(4, 8, 4);

    // 3. 需要 Padding/Masking：N 不是 NumPEs 的倍数
    // Matrix 4x7, PE=4 (第一批4个，第二批3个+1空闲)
    run_test(4, 7, 4);

    // 4. 更大的输入维度 (K) 测试累加准确性
    // Matrix 32x4, PE=4
    run_test(32, 4, 4);

    // 5. 混合情况：较大规模随机测试
    // Matrix 64x128, PE=16
    run_test(64, 128, 16);
    
    // 6. 极小情况
    run_test(1, 1, 1);

    return 0;
}
