#pragma once
#include <vector>
#include <deque>
#include <cstdint>
#include <iostream>
#include <string>
#include "Memory/HBMController.h"
#include "Memory/DDRController.h"

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

    int seq_len;
    int selected_kv_len;
    
    // Timestamps for logging (Gantt Chart)
    uint64_t p_start = 0, p_end = 0;
    uint64_t f_start = 0, f_end = 0;
    uint64_t c_start = 0, c_end = 0;
    
    Stage current_stage = Stage::IDLE;
};

// Configuration for the Simulator
struct PipelineConfig {
    // System Params
    double frequency_mhz;   // e.g., 200.0 for 200MHz
    size_t max_queue_size = 2; // Finite FIFO depth for backpressure simulation

    // Model Params
    uint64_t context_length;
    uint64_t head_dim;         // e.g., 128
    uint64_t bytes_per_elem;   // e.g., 2 for BF16
    double sparsity_ratio;     // alpha
    uint64_t num_head_groups;
    uint64_t heads_per_group;
    
    // Hardware Params
    // P-Stage
    double hbm_scan_bandwidth_gbps; 
    uint64_t int4_vector_size_bytes; // Size of compressed Int4 vector per token per head
    
    // C-Stage
    uint64_t num_bf16_pes; // e.g., 64 (Realistic FPGA Resource)
    
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
    
    uint64_t get_total_cycles() const { return current_cycle; }
    uint64_t get_total_bubbles() const { return total_bubbles; }

private:
    PipelineConfig config;
    Memory::DDRController& ddr;
    
    uint64_t current_cycle = 0;
    uint64_t total_bubbles = 0;
    
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
    
    uint64_t p_stage_free_cycle = 0;
    uint64_t f_stage_free_cycle = 0;
    uint64_t c_stage_free_cycle = 0;

    // Scaling factors for F-Stage extrapolation
    uint64_t f_num_reqs = 0;
    uint64_t f_sim_count = 0;

    // Helper functions
    uint64_t calculate_p_latency(const HeadGroupTask& task);
    uint64_t calculate_c_latency(const HeadGroupTask& task);
    // F-Latency is calculated dynamically via DDRController
    
    void step();
};

} // namespace System
