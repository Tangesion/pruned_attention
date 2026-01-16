#include "PE/scheduler/GemvScheduler.h"
#include "PE/units/ReduceMacArrayUnit.h"
#include "bf16/add_sim.h"
#include "bf16/multiply_sim.h"
#include "bf16/bf16_basic_ops.h"
#include "int4/multiply_sim.h"
#include "int4/add_sim.h"
#include "int4/int4_basic_ops.h"
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

// Helper to generate random int4 vector (stored as int8_t for convenience, range [-8, 7])
std::vector<int8_t> generate_random_int4_vector(size_t size) {
    std::vector<int8_t> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(-8, 7);
    for (size_t i = 0; i < size; ++i) {
        vec[i] = static_cast<int8_t>(dis(gen));
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

// Reference GEMV implementation (int32)
std::vector<int32_t> reference_gemv_int(const std::vector<std::vector<int8_t>>& matrix, const std::vector<int8_t>& input) {
    if (matrix.empty() || input.empty()) return {};
    size_t K = input.size();
    size_t N = matrix[0].size();
    std::vector<int32_t> output(N, 0);

    for (size_t n = 0; n < N; ++n) {
        for (size_t k = 0; k < K; ++k) {
            output[n] += static_cast<int32_t>(input[k]) * static_cast<int32_t>(matrix[k][n]);
        }
    }
    return output;
}

void run_test_multibatch_bf16(size_t B, size_t K, size_t N, size_t num_pes, bool expect_packed) {
    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "BF16 Test Configuration: Batches=" << B << ", Input(K)=" << K << ", Output(N)=" << N << ", Num PEs=" << num_pes << std::endl;
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

void run_test_multibatch_int4(size_t B, size_t K, size_t N, size_t num_pes, bool expect_packed) {
    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "INT4 Test Configuration: Batches=" << B << ", Input(K)=" << K << ", Output(N)=" << N << ", Num PEs=" << num_pes << std::endl;
    std::cout << "Expected Mode: " << (expect_packed ? "Packed (Parallel)" : "Sequential (Tiled)") << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    // 1. Generate Data
    std::vector<std::vector<int8_t>> inputs_int(B);
    std::vector<std::vector<std::vector<int8_t>>> matrices_int(B);
    std::vector<std::vector<int32_t>> expected_int(B);

    std::vector<std::vector<uint16_t>> inputs_sim(B);
    std::vector<std::vector<std::vector<uint16_t>>> matrices_sim(B);

    for (size_t b = 0; b < B; ++b) {
        inputs_int[b] = generate_random_int4_vector(K);
        matrices_int[b].resize(K);
        for (size_t k = 0; k < K; ++k) {
            matrices_int[b][k] = generate_random_int4_vector(N);
        }
        expected_int[b] = reference_gemv_int(matrices_int[b], inputs_int[b]);

        // Convert to Simulator Format (uint16_t low 4 bits)
        inputs_sim[b].resize(K);
        for (size_t k = 0; k < K; ++k) {
            inputs_sim[b][k] = static_cast<uint16_t>(inputs_int[b][k] & 0xF);
        }

        matrices_sim[b].resize(K);
        for (size_t k = 0; k < K; ++k) {
            matrices_sim[b][k].resize(N);
            for (size_t n = 0; n < N; ++n) {
                matrices_sim[b][k][n] = static_cast<uint16_t>(matrices_int[b][k][n] & 0xF);
            }
        }
    }

    // 2. Setup Scheduler
    auto mult_pipe_factory = []() { return std::make_unique<int4::Int4MultiplyPipeline>(1); };
    auto add_pipe_factory = []() { return std::make_unique<int4::Int4AddPipeline>(1); };
    auto mac_array = std::make_unique<ReduceMacArrayUnit>(mult_pipe_factory, add_pipe_factory, num_pes);

    GemvScheduler::Config config{32.0, num_pes};
    GemvScheduler scheduler(std::move(mac_array), config);

    // 3. Run Simulation
    std::cout << "Running simulation..." << std::endl;
    auto results_int32 = scheduler.run_gemv_int32(matrices_sim, inputs_sim);
    
    size_t total_cycles = scheduler.get_total_cycles();
    size_t total_ops = B * K * N; 
    double utilization = (double(total_ops) / (total_cycles * num_pes)) * 100.0;

    std::cout << "Performance Stats:" << std::endl;
    std::cout << "  Total Cycles: " << total_cycles << std::endl;
    std::cout << "  Total MAC Ops: " << total_ops << std::endl;
    std::cout << "  Utilization: " << std::fixed << std::setprecision(2) << utilization << "%" << std::endl;
    
    // 4. Verify Results
    if (results_int32.size() != B) {
        std::cerr << "Error: Output batch size mismatch. Expected " << B << ", got " << results_int32.size() << std::endl;
        return;
    }

    bool passed = true;
    for (size_t b = 0; b < B; ++b) {
        if (results_int32[b].size() != N) {
            std::cerr << "Error: Output size mismatch in batch " << b << std::endl;
            return;
        }
        for (size_t i = 0; i < N; ++i) {
            if (results_int32[b][i] != expected_int[b][i]) {
                passed = false;
                std::cout << "Mismatch at Batch " << b << ", Index " << i 
                          << ": Expected " << expected_int[b][i] 
                          << ", Got " << results_int32[b][i] << std::endl;
            }
        }
    }

    if (passed) {
         std::cout << "  Status: PASSED" << std::endl;
    } else {
         std::cout << "  Status: FAILED" << std::endl;
    }
}

int main() {
    // === BF16 Tests ===
    std::cout << "================ BF16 Tests ================" << std::endl;
    // 1. Packed Mode Test
    run_test_multibatch_bf16(2, 32, 8, 16, true);

    // 2. Sequential Mode Test
    run_test_multibatch_bf16(2, 32, 16, 16, false);

    // 3. Mixed/Small N Test (Packed)
    run_test_multibatch_bf16(3, 32, 4, 16, true);

    // 4. Large Sequential Test
    run_test_multibatch_bf16(2, 64, 64, 64, false);

    // 5. llama q * kt
    run_test_multibatch_bf16(2, 128, 1024, 64, false);

    // 6. MHA Output Phase
    run_test_multibatch_bf16(32, 128, 128, 1024, true);

    run_test_multibatch_bf16(32, 128, 1024 * 3, 512, true);

    // === INT4 Tests ===
    std::cout << "\n================ INT4 Tests ================" << std::endl;
    // Same configurations but checking exact integer results
    
    // 1. Packed Mode Test
    run_test_multibatch_int4(2, 32, 8, 16, true);

    // 2. Sequential Mode Test
    run_test_multibatch_int4(2, 32, 16, 16, false);

    // 6. MHA Output Phase (Chunked Parallelism)
    run_test_multibatch_int4(32, 128, 1024 * 3, 512 * 4, true);

    return 0;
}
