#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include "System/PipelineSimulator.h"
#include "Memory/DDRController.h"

using namespace System;
using namespace Memory;

void run_experiment(int num_channels, const std::string& label) {
    std::cout << "\n=================================================================\n";
    std::cout << " Running Experiment: " << label << " (" << num_channels << " Channels)\n";
    std::cout << "=================================================================\n";

    // 1. Setup DDR Controller
    DDRController::Config ddr_cfg;
    ddr_cfg.tCL = 14;
    ddr_cfg.tRCD = 14;
    ddr_cfg.tRP = 14;
    ddr_cfg.tBURST = 4;
    ddr_cfg.num_channels = num_channels; // <--- Dynamic
    ddr_cfg.num_banks_per_channel = 16;
    ddr_cfg.row_size_bytes = 8192;
    ddr_cfg.mapping_strategy = BankMappingStrategy::HASH_XOR;
    
    DDRController ddr(ddr_cfg);
    
    // 2. Setup Pipeline Config
    PipelineConfig pipe_cfg;
    pipe_cfg.frequency_mhz = 200.0;
    pipe_cfg.context_length = 32 * 1024;
    pipe_cfg.head_dim = 128;
    pipe_cfg.bytes_per_elem = 2;
    pipe_cfg.sparsity_ratio = 0.05;
    pipe_cfg.num_head_groups = 8;
    pipe_cfg.heads_per_group = 4;
    pipe_cfg.hbm_scan_bandwidth_gbps = 400.0;
    pipe_cfg.int4_vector_size_bytes = 64;
    pipe_cfg.num_bf16_pes = 512; 
    pipe_cfg.P_stage_pe_nums = 2048;
    pipe_cfg.C_stage_pe_nums = 256;
    pipe_cfg.max_queue_size = 8;

    // TFU & URAM Config
    pipe_cfg.uram_bandwidth_gbps = 1000.0; // URAM is on-chip, very fast
    pipe_cfg.uram_latency_cycles = 5;
    pipe_cfg.tfu_throughput_per_cycle = 64; // Can filter 64 tokens per cycle

    // 3. Run
    PipelineSimulator sim(pipe_cfg, ddr);

    sim.run();
    
    // 4. Analysis
    auto logs = sim.get_task_logs();
    size_t total_cycles = sim.get_total_cycles();
    
    // Calculate naive serial latency
    size_t serial_latency = 0;
    size_t total_p = 0, total_f = 0, total_c = 0;
    
    std::cout << std::left 
          << std::setw(10) << "Group" 
          << std::setw(15) << "P-Stage" 
          << std::setw(15) << "F-Stage" 
          << std::setw(15) << "C-Stage" << std::endl;

    for (const auto& t : logs) {
        size_t p = t.p_end - t.p_start;
        size_t f = t.f_end - t.f_start;
        size_t c = t.c_end - t.c_start;
        serial_latency += p + f + c;
        total_p += p; total_f += f; total_c += c;
        
        std::cout << std::left 
                  << std::setw(10) << ("G" + std::to_string(t.group_id))
                  << std::setw(15) << p
                  << std::setw(15) << f
                  << std::setw(15) << c << std::endl;
    }
    
    // Average Stage Latency
    double avg_p = (double)total_p / logs.size();
    double avg_f = (double)total_f / logs.size();
    double avg_c = (double)total_c / logs.size();
    
    std::cout << "\n[Metrics]\n";
    std::cout << "Avg Stage Latency: P=" << (int)avg_p << ", F=" << (int)avg_f << ", C=" << (int)avg_c << std::endl;
    std::cout << "Total Latency:     " << total_cycles << " Cycles\n";
    std::cout << "Serial Latency:    " << serial_latency << " Cycles\n";
    std::cout << "Speedup:           " << std::fixed << std::setprecision(2) << (double)serial_latency / total_cycles << "x\n";
    
    size_t bubbles = sim.get_total_bubbles();
    std::cout << "Bubble Rate:       " << ((double)bubbles / total_cycles * 100.0) << "%\n";
    
    // Bottleneck Analysis
    std::string bottleneck = "Balanced";
    if (avg_f > avg_p && avg_f > avg_c) bottleneck = "Memory Bound (DDR)";
    else if (avg_c > avg_p && avg_c > avg_f) bottleneck = "Compute Bound (BF16)";
    else if (avg_p > avg_f && avg_p > avg_c) bottleneck = "Bandwidth Bound (HBM)";
    
    std::cout << "System State:      " << bottleneck << std::endl;
}

int main() {
    run_experiment(8, "Baseline");
    run_experiment(16, "Hardware Optimization");
    return 0;
}