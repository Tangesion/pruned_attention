#include "PE/scheduler/GemvScheduler.h"
#include "PE/units/ReduceMacArrayUnit.h"
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

void run_test_multibatch(size_t B, size_t K, size_t N, size_t num_pes, bool expect_packed) {
    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "Test Configuration: Batches=" << B << ", Input(K)=" << K << ", Output(N)=" << N << ", Num PEs=" << num_pes << std::endl;
    std::cout << "Expected Mode: " << (expect_packed ? "Packed (Parallel)" : "Sequential (Tiled)") << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    // 1. Generate Data
    std::vector<std::vector<float>> inputs_f(B);
    std::vector<std::vector<std::vector<float>>> matrices_f(B);
    std::vector<std::vector<float>> expected_f(B);

    std::vector<std::vector<uint16_t>> inputs_bf16(B);
    std::vector<std::vector<std::vector<uint16_t>>> matrices_bf16(B);

    for (size_t b = 0; b < B; ++b) {
        inputs_f[b] = generate_random_vector(K);
        matrices_f[b].resize(K);
        for (size_t k = 0; k < K; ++k) {
            matrices_f[b][k] = generate_random_vector(N);
        }
        expected_f[b] = reference_gemv(matrices_f[b], inputs_f[b]);

        inputs_bf16[b].resize(K);
        for (size_t k = 0; k < K; ++k) {
            inputs_bf16[b][k] = bf16::float_to_bf16(inputs_f[b][k]);
        }

        matrices_bf16[b].resize(K);
        for (size_t k = 0; k < K; ++k) {
            matrices_bf16[b][k].resize(N);
            for (size_t n = 0; n < N; ++n) {
                matrices_bf16[b][k][n] = bf16::float_to_bf16(matrices_f[b][k][n]);
            }
        }
    }

    // 2. Setup Scheduler
    auto mult_pipe_factory = []() { return std::make_unique<bf16::BF16MultiplyPipeline>(); };
    auto add_pipe_factory = []() { return std::make_unique<bf16::BF16AddPipeline>(); };
    auto mac_array = std::make_unique<ReduceMacArrayUnit>(mult_pipe_factory, add_pipe_factory, num_pes);

    GemvScheduler::Config config{32.0, num_pes};
    GemvScheduler scheduler(std::move(mac_array), config);

    // 3. Run Simulation
    std::cout << "Running simulation..." << std::endl;
    auto results_bf16 = scheduler.run_gemv(matrices_bf16, inputs_bf16);
    
    size_t total_cycles = scheduler.get_total_cycles();
    size_t total_ops = B * K * N; // MAC operations
    double utilization = (double(total_ops) / (total_cycles * num_pes)) * 100.0;

    std::cout << "Performance Stats:" << std::endl;
    std::cout << "  Total Cycles: " << total_cycles << std::endl;
    std::cout << "  Total MAC Ops: " << total_ops << std::endl;
    std::cout << "  Utilization: " << std::fixed << std::setprecision(2) << utilization << "%" << std::endl;
    
    // 4. Verify Results
    if (results_bf16.size() != B) {
        std::cerr << "Error: Output batch size mismatch. Expected " << B << ", got " << results_bf16.size() << std::endl;
        return;
    }

    float max_error = 0.0f;
    float total_error = 0.0f;

    for (size_t b = 0; b < B; ++b) {
        if (results_bf16[b].size() != N) {
            std::cerr << "Error: Output size mismatch in batch " << b << std::endl;
            return;
        }
        for (size_t i = 0; i < N; ++i) {
            float val = bf16::bf16_to_float(results_bf16[b][i]);
            float expected = expected_f[b][i];
            float error = std::abs(val - expected);
            total_error += error;
            max_error = std::max(max_error, error);
        }
    }

    std::cout << "Verification Result:" << std::endl;
    std::cout << "  Max Absolute Error: " << max_error << std::endl;
    std::cout << "  Avg Absolute Error: " << total_error / (B * N) << std::endl;

    if (max_error > 2.0f) { 
         std::cout << "  Status: WARNING / FAIL (High Error)" << std::endl;
    } else {
         std::cout << "  Status: PASSED" << std::endl;
    }
}

int main() {
    // 1. Packed Mode Test
    // B=2, N=8, PEs=16. (2*8 = 16 <= 16). Perfect packing.
    run_test_multibatch(2, 32, 8, 16, true);

    // 2. Sequential Mode Test
    // B=2, N=16, PEs=16. (2*16 = 32 > 16). Must be sequential.
    run_test_multibatch(2, 32, 16, 16, false);

    // 3. Mixed/Small N Test (Packed)
    // B=3, N=4, PEs=16. (3*4 = 12 <= 16). 
    run_test_multibatch(3, 32, 4, 16, true);

    // 4. Large Sequential Test (Folding + Sequential)
    // B=2, N=32, PEs=16. (Batch 0: 2 tiles, Batch 1: 2 tiles)
    run_test_multibatch(2, 64, 64, 64, false);

    // 5. llama q * kt
    // B=2, N=1024, K=128，PEs=64 -> Sequential
    run_test_multibatch(2, 128, 1024, 64, false);

    // 6. MHA Output Phase (Chunked Parallelism)
    // B=32 Heads, N=128 (Head Dim), PEs=1024.
    // 1024 / 128 = 8 batches per chunk.
    // Total 4 chunks (32/8).
    run_test_multibatch(32, 128, 128, 1024, true);

    return 0;
}
