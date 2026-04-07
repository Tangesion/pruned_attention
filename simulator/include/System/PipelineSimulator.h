#pragma once
#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>
#include <deque>
#include <cstdint>
#include <iostream>
#include <string>
#include "Memory/DDRController.h"
#include "Memory/HBMController.h"
#include "PE/scheduler/GemvScheduler.h"
#include "PE/units/ReduceMacArrayUnit.h"
#include "bf16/add_sim.h"
#include "bf16/multiply_sim.h"
#include "bf16/bf16_basic_ops.h"
#include "int4/multiply_sim.h"
#include "int4/add_sim.h"
#include "int4/int4_basic_ops.h"
#include "System/utils.h"

namespace System {

// Pipeline Stages
enum class Stage {
    IDLE,
    PREDICT_HBM,    // P-Stage: Int4 Scan (HBM Bound)
    FETCH_DDR,      // F-Stage: Sparse Fetch (DDR Bound)
    COMPUTE_BF16,   // C-Stage: Attention Compute (Compute Bound)
    FINISHED
};

// Represents a single Head Group processing task
struct HeadGroupTask {
    int group_id;
    int num_heads;

    size_t seq_len;
    size_t selected_kv_len;

    size_t P_stage_pe_nums;
    size_t C_stage_pe_nums;
    
    // Timestamps for logging (Gantt Chart)
    size_t p_start = 0, p_end = 0;
    size_t f_start = 0, f_end = 0;
    size_t c_start = 0, c_end = 0;
    
    Stage current_stage = Stage::IDLE;
};

// Configuration for the Simulator
struct PipelineConfig {
    // System Params
    double frequency_mhz;   // e.g., 200.0 for 200MHz
    size_t max_queue_size = 2; // Finite FIFO depth for backpressure simulation
    bool verbose_logging = false;
    bool use_realistic_stage_models = true;
    bool use_hash_aware_fetch = false;
    bool use_dual_fetch = false;
    size_t fetch_num_lanes = 1;
    size_t max_sgu_merge_bytes = 4096;
    size_t fetch_max_inflight_bursts = 256;
    size_t fetch_head_stride_padding_bytes = 0;

    // Model Params
    size_t context_length;
    size_t head_dim;         // e.g., 128
    size_t bytes_per_elem;   // e.g., 2 for BF16
    double sparsity_ratio;     // alpha
    size_t num_head_groups;
    size_t heads_per_group;

    size_t selected_kv_len;
    
    // Hardware Params
    // P-Stage
    double hbm_scan_bandwidth_gbps;
    size_t int4_vector_size_bytes; // Size of compressed Int4 vector per token per head
    size_t P_stage_pe_nums;

    // TFU & URAM (Filtering)
    double uram_bandwidth_gbps;     // e.g., 600.0 (High Bandwidth on-chip)
    size_t uram_latency_cycles;     // Fixed access latency
    size_t tfu_throughput_per_cycle; // How many tokens can be filtered per cycle

    // C-Stage
    size_t num_bf16_pes; // e.g., 64 (Realistic FPGA Resource)
    size_t C_stage_pe_nums;
    
    // F-Stage (Latency comes from DDRController)
    // We assume S_blk (512B) aggregation happens in hardware
};

class PipelineSimulator {
public:
    PipelineSimulator(PipelineConfig config, Memory::DDRController& ddr_controller);

    // Run the simulation until all head groups are finished
    void run();

    // Get the trace logs for visualization
    const std::vector<HeadGroupTask>& get_task_logs() const;

    size_t get_total_cycles() const { return current_cycle; }
    size_t get_total_bubbles() const { return total_bubbles; }

private:
    PipelineConfig config;
    Memory::DDRController& ddr;
    Memory::HBMController hbm; // HBM Controller for P-Stage

    std::unique_ptr<PE::GemvScheduler> predict_gemv_scheduler;
    std::unique_ptr<PE::GemvScheduler> compute_gemv_scheduler;
    
    size_t current_cycle = 0;
    size_t total_bubbles = 0;
    
    // All tasks to process
    std::vector<HeadGroupTask> tasks;
    
    // Queues between stages
    std::deque<int> pending_start_queue; // Waiting for P-Stage
    std::deque<int> pending_fetch_queue; // Finished P, Waiting for F-Stage
    std::deque<int> pending_compute_queue;// Finished F, Waiting for C-Stage
    
    // Resource Status
    // -1 means idle, >=0 means processing task ID
    int busy_p_stage_task = -1;
    int busy_f_stage_task = -1;
    int busy_c_stage_task = -1;
    
    size_t p_stage_free_cycle = 0;
    size_t f_stage_free_cycle = 0;
    size_t c_stage_free_cycle = 0;

    // Helper functions
    size_t calculate_p_latency(const HeadGroupTask& task);
    size_t calculate_f_latency(const HeadGroupTask& task);
    size_t calculate_c_latency(const HeadGroupTask& task);
    // F-Latency is calculated dynamically via DDRController

    size_t calculate_p_latency_realistic(const HeadGroupTask& task);
    size_t calculate_c_latency_realistic(const HeadGroupTask& task);
    
    void step();
};

} // namespace System
