#include "PE/scheduler/GemvScheduler.h"
#include "PE/units/ReduceMacArrayUnit.h"
#include "bf16/add_sim.h"
#include "bf16/multiply_sim.h"
#include "bf16/bf16_basic_ops.h"
#include "int4/add_sim.h"
#include "int4/multiply_sim.h"
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

struct PerfStats {
    size_t total_cycles;
    size_t total_ops;
    double utilization;
};

// Run BF16 Simulation
PerfStats run_bf16_sim(size_t K, size_t N, size_t num_pes, 
                       const std::vector<std::vector<uint16_t>>& matrix, 
                       const std::vector<uint16_t>& input) {
    
    auto mult_pipe_factory = []() { return std::make_unique<bf16::BF16MultiplyPipeline>(); };
    auto add_pipe_factory = []() { return std::make_unique<bf16::BF16AddPipeline>(); };
    auto mac_array = std::make_unique<ReduceMacArrayUnit>(mult_pipe_factory, add_pipe_factory, num_pes);

    GemvScheduler::Config config{1000.0, num_pes}; // Unlimited bandwidth
    GemvScheduler scheduler(std::move(mac_array), config);

    scheduler.run_gemv(matrix, input);
    
    size_t total_cycles = scheduler.get_total_cycles();
    size_t total_ops = K * N;
    double utilization = (double(total_ops) / (total_cycles * num_pes)) * 100.0;
    
    return {total_cycles, total_ops, utilization};
}

// Run Int4 Simulation (Using same input data container but interpreted as Int4 simulation)
// Note: We use the same BF16 u16 data as input placeholders because run_gemv_int32 takes u16.
// In a real scenario, data packing would differ, but for cycle accuracy, it's fine.
PerfStats run_int4_sim(size_t K, size_t N, size_t num_pes, 
                       const std::vector<std::vector<uint16_t>>& matrix, 
                       const std::vector<uint16_t>& input) {
    
    auto mult_pipe_factory = []() { return std::make_unique<int4::Int4MultiplyPipeline>(); };
    auto add_pipe_factory = []() { return std::make_unique<int4::Int4AddPipeline>(); };
    auto mac_array = std::make_unique<ReduceMacArrayUnit>(mult_pipe_factory, add_pipe_factory, num_pes);

    GemvScheduler::Config config{1000.0, num_pes};
    GemvScheduler scheduler(std::move(mac_array), config);

    scheduler.run_gemv_int32(matrix, input);
    
    size_t total_cycles = scheduler.get_total_cycles();
    size_t total_ops = K * N;
    double utilization = (double(total_ops) / (total_cycles * num_pes)) * 100.0;
    
    return {total_cycles, total_ops, utilization};
}

void run_comparison_test(size_t K, size_t N, size_t bf16_pes) {
    size_t int4_pes = bf16_pes * 4; // Assume 4x density for Int4

    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "Comparison Test: Input(K)=" << K << ", Output(N)=" << N << std::endl;
    std::cout << "  BF16 PEs: " << bf16_pes << std::endl;
    std::cout << "  Int4 PEs: " << int4_pes << " (2x Density)" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Dummy Data (Values don't matter for cycle count)
    std::vector<uint16_t> input(K, 0);
    std::vector<std::vector<uint16_t>> matrix(K, std::vector<uint16_t>(N, 0));

    // Run BF16
    auto bf16_stats = run_bf16_sim(K, N, bf16_pes, matrix, input);
    
    // Run Int4
    auto int4_stats = run_int4_sim(K, N, int4_pes, matrix, input);

    // Report
    std::cout << std::left << std::setw(15) << "Metric" 
              << std::setw(15) << "BF16" 
              << std::setw(15) << "Int4" 
              << "Improvement" << std::endl;
    
    std::cout << std::string(60, '.') << std::endl;

    std::cout << std::left << std::setw(15) << "Total Cycles" 
              << std::setw(15) << bf16_stats.total_cycles 
              << std::setw(15) << int4_stats.total_cycles 
              << std::fixed << std::setprecision(2) << (double)bf16_stats.total_cycles / int4_stats.total_cycles << "x" << std::endl;

    std::cout << std::left << std::setw(15) << "Utilization" 
              << std::fixed << std::setprecision(2) 
              << bf16_stats.utilization << "%      " 
              << int4_stats.utilization << "%" << std::endl;
}

int main() {
    // 1. Small Case
    run_comparison_test(64, 64, 16);

    // 2. Medium Case (N aligned with BF16, unaligned with Int4?)
    // N=128. BF16=16 -> 8 passes. Int4=32 -> 4 passes. Expected 2x.
    run_comparison_test(128, 128, 16);

    // 3. Large Case
    run_comparison_test(1024, 1024, 64);

    // 4. Odd Size
    run_comparison_test(511, 255, 32);

    return 0;
}