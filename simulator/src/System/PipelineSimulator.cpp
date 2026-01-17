#include "System/PipelineSimulator.h"
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <set>
#include <vector>

namespace System {

PipelineSimulator::PipelineSimulator(PipelineConfig config, Memory::DDRController& ddr_controller)
    : config(config), ddr(ddr_controller) {
    
    // Initialize tasks
    for (int i = 0; i < config.num_head_groups; ++i) {
        HeadGroupTask task;
        task.group_id = i;
        task.num_heads = config.heads_per_group;
        task.seq_len = config.context_length;
        task.selected_kv_len = (size_t)(config.context_length * config.sparsity_ratio);
        task.P_stage_pe_nums = config.P_stage_pe_nums;
        task.C_stage_pe_nums = config.C_stage_pe_nums;
        task.current_stage = Stage::IDLE;
        tasks.push_back(task);
        pending_start_queue.push_back(i);
    }

    auto predict_mult_pipe_factory = []() { return std::make_unique<int4::Int4MultiplyPipeline>(1); };
    auto predict_add_pipe_factory = []() { return std::make_unique<int4::Int4AddPipeline>(1); };
    auto predict_mac_array = std::make_unique<PE::ReduceMacArrayUnit>(predict_mult_pipe_factory, predict_add_pipe_factory, config.P_stage_pe_nums);
    PE::GemvScheduler::Config predict_config{ config.P_stage_pe_nums };
    predict_gemv_scheduler = std::make_unique<PE::GemvScheduler>(std::move(predict_mac_array), predict_config);

    auto compute_mult_pipe_factory = []() { return std::make_unique<bf16::BF16MultiplyPipeline>(); };
    auto compute_add_pipe_factory = []() { return std::make_unique<bf16::BF16AddPipeline>(); };
    auto compute_mac_array = std::make_unique<PE::ReduceMacArrayUnit>(compute_mult_pipe_factory, compute_add_pipe_factory, config.num_bf16_pes);
    PE::GemvScheduler::Config compute_config{ config.C_stage_pe_nums };
    compute_gemv_scheduler = std::make_unique<PE::GemvScheduler>(std::move(compute_mac_array), compute_config);

}

size_t PipelineSimulator::calculate_p_latency(const HeadGroupTask& task) {
    // Formula: T_pred = L_ctx * S_vec * M / BW_scan
    
    double bytes_total = (double)config.context_length * config.int4_vector_size_bytes * task.num_heads;
    
    // 1. Calculate Time in nanoseconds
    // Bandwidth is in GB/s. Note: 1 GB/s = 1 Byte/ns (approx, using 10^9 definition)
    double time_ns = bytes_total / config.hbm_scan_bandwidth_gbps;
    
    // 2. Calculate Clock Period in nanoseconds
    // Period = 1 / f. If f is in MHz, Period(ns) = 1000 / MHz
    double period_ns = 1000.0 / config.frequency_mhz;
    
    // 3. Convert to Cycles
    size_t cycles = static_cast<size_t>(std::ceil(time_ns / period_ns));
    
    return std::max((size_t)1, cycles);
}

size_t PipelineSimulator::calculate_c_latency(const HeadGroupTask& task) {
    // Formula: T_comp = Total MACs / Ops_Per_Cycle
    // We align with the detailed GEMV simulator logic (Cycle-Accurate)
    
    size_t num_tokens = static_cast<size_t>(config.context_length * config.sparsity_ratio);
    
    // Total MAC Operations = Tokens * Head_Dim * Num_Heads
    // Each token requires a dot product of size Head_Dim with the Query
    // This happens for each head in the group.
    size_t total_macs = num_tokens * config.head_dim * task.num_heads;
    
    // Hardware Throughput = Num PEs (1 MAC per cycle per PE)
    size_t ops_per_cycle = config.num_bf16_pes;
    
    if (ops_per_cycle == 0) return 1; // Safety check
    
    // Basic cycles + Pipeline Fill/Drain Overhead (e.g., ~50 cycles)
    size_t cycles = (total_macs / ops_per_cycle) + 50;
    
    return std::max((size_t)1, cycles);
}

size_t PipelineSimulator::calculate_p_latency_realistic(const HeadGroupTask& task) {
    // score = q * k
    size_t num_heads = task.num_heads;
    size_t seq_len = task.seq_len;
    size_t head_dim = config.head_dim;

    std::vector<std::vector<uint16_t>> q(num_heads);
    std::vector<std::vector<std::vector<uint16_t>>> k(num_heads);

    System::generate_random_int4_qk(num_heads, seq_len, head_dim, q, k);

    auto result = predict_gemv_scheduler->run_gemv_int32(k, q);
    return predict_gemv_scheduler->get_total_cycles();

}

size_t PipelineSimulator::calculate_c_latency_realistic(const HeadGroupTask& task) {
    // attn_output = softmax(score) * v
    size_t num_heads = task.num_heads;
    size_t selected_kv_len = task.selected_kv_len;
    size_t head_dim = config.head_dim;

    std::vector<std::vector<uint16_t>> q(num_heads);
    std::vector<std::vector<std::vector<uint16_t>>> k(num_heads);
    std::vector<std::vector<std::vector<uint16_t>>> v(num_heads);

    //std::vector<std::vector<uint16_t>> score(num_heads)
    System::generate_random_bf16_qkv(num_heads, selected_kv_len, head_dim, q, k, v);
    //System::generate_random_bf16_score(num_heads, selected_kv_len, score);
    

    auto result1 = compute_gemv_scheduler->run_gemv(k, q);
    auto result2 = compute_gemv_scheduler->run_gemv(v, result1);
    return compute_gemv_scheduler->get_total_cycles();
}


size_t per_group_p_latency = 0;
size_t per_group_c_latency = 0;


void PipelineSimulator::step() {
    bool p_active = false;
    bool f_active = false;
    bool c_active = false;

    // --- C-Stage: BF16 Attention Compute ---
    if (busy_c_stage_task != -1) {
        c_active = true;
        if (current_cycle >= c_stage_free_cycle) {
            tasks[busy_c_stage_task].c_end = current_cycle;
            tasks[busy_c_stage_task].current_stage = Stage::FINISHED;
            busy_c_stage_task = -1;
        }
    }
    
    if (busy_c_stage_task == -1 && !pending_compute_queue.empty()) {
        int task_id = pending_compute_queue.front();
        pending_compute_queue.pop_front();
        busy_c_stage_task = task_id;
        c_active = true;
        
        tasks[task_id].current_stage = Stage::COMPUTE_BF16;
        tasks[task_id].c_start = current_cycle;
        //c_stage_free_cycle = current_cycle + calculate_c_latency(tasks[task_id]);
        if (per_group_c_latency == 0) {
            per_group_c_latency = calculate_c_latency_realistic(tasks[task_id]);
            std::cout << "Per-Group C-Stage Latency (Realistic): " << per_group_c_latency << " cycles\n";
        }
        c_stage_free_cycle = current_cycle + per_group_c_latency;
    }

    // --- F-Stage: DDR Sparse KV Fetch ---
    if (busy_f_stage_task != -1) {
        f_active = true;
        // Check Backpressure: Can we push to Compute Queue?
        if (current_cycle >= f_stage_free_cycle) {
            if (pending_compute_queue.size() < config.max_queue_size) {
                tasks[busy_f_stage_task].f_end = current_cycle;
                pending_compute_queue.push_back(busy_f_stage_task);
                busy_f_stage_task = -1;
            } else {
                // STALL: Hold resource, do not free busy_f_stage_task
            }
        }
    }

    if (busy_f_stage_task == -1 && !pending_fetch_queue.empty()) {
        int task_id = pending_fetch_queue.front();
        pending_fetch_queue.pop_front();
        busy_f_stage_task = task_id;
        f_active = true;
        
        tasks[task_id].current_stage = Stage::FETCH_DDR;
        tasks[task_id].f_start = current_cycle;
        
        size_t token_kv_size = config.head_dim * config.bytes_per_elem * 2; 
        size_t num_selected = static_cast<size_t>(config.context_length * config.sparsity_ratio);
        
        f_num_reqs = num_selected * tasks[busy_f_stage_task].num_heads;
        f_sim_count = std::min(f_num_reqs, (size_t)500); 

        std::mt19937 gen(1234 + task_id);
        std::uniform_int_distribution<size_t> dist(0, config.context_length - 1);

        //for (size_t i = 0; i < f_sim_count; ++i) {
        //    size_t token_idx = dist(gen);
        //    size_t addr = token_idx * token_kv_size;
        //    ddr.send_request(Memory::MemoryRequest(i, addr, token_kv_size, Memory::RequestType::READ, current_cycle));
        //}

        // --- SGU-like optimization: sort (ROB) + merge adjacent token requests ---
        std::vector<size_t> token_indices;
        token_indices.reserve(f_sim_count);
        for (size_t i = 0; i < f_sim_count; ++i) {
            token_indices.push_back(dist(gen));
        }
        std::sort(token_indices.begin(), token_indices.end());

        size_t id_counter = 0;
        if (!token_indices.empty()) {
            size_t current_start_idx = token_indices[0];
            size_t current_count = 1;

            for (size_t i = 1; i < token_indices.size(); ++i) {
                // Adjacent token -> merge into current burst
                if (token_indices[i] == current_start_idx + current_count) {
                    current_count++;
                } else {
                    // Flush current burst
                    size_t addr = current_start_idx * token_kv_size;
                    size_t bytes = current_count * token_kv_size;
                    ddr.send_request(Memory::MemoryRequest(
                        id_counter++,
                        addr,
                        bytes,
                        Memory::RequestType::READ,
                        current_cycle
                    ));

                    // Start new burst
                    current_start_idx = token_indices[i];
                    current_count = 1;
                }
            }

            // Flush last burst
            {
                size_t addr = current_start_idx * token_kv_size;
                size_t bytes = current_count * token_kv_size;
                ddr.send_request(Memory::MemoryRequest(
                    id_counter++,
                    addr,
                    bytes,
                    Memory::RequestType::READ,
                    current_cycle
                ));
            }
        }
        f_stage_free_cycle = UINT64_MAX; // Mark as waiting for DDR sample
    }
    
    // Poll DDR completion and Extrapolate
    if (busy_f_stage_task != -1 && f_stage_free_cycle == UINT64_MAX) {
        ddr.step(current_cycle);
        if (ddr.is_idle()) {
            // Sample finished. Extrapolate total time.
            size_t sample_lat = current_cycle - tasks[busy_f_stage_task].f_start;
            double scale_factor = (double)f_num_reqs / f_sim_count;
            size_t total_est_lat = static_cast<size_t>(sample_lat * scale_factor);
            
            // Update free cycle to the future estimated time
            f_stage_free_cycle = tasks[busy_f_stage_task].f_start + total_est_lat;
            
            // Corner case: if total_est_lat is smaller than current sample_lat (shouldn't happen with scale >= 1)
            if (f_stage_free_cycle < current_cycle) f_stage_free_cycle = current_cycle;
        }
    }

    // --- P-Stage: Int4 HBM Scan ---
    if (busy_p_stage_task != -1) {
        p_active = true;
        // Check Backpressure: Can we push to Fetch Queue?
        if (current_cycle >= p_stage_free_cycle) {
            if (pending_fetch_queue.size() < config.max_queue_size) {
                tasks[busy_p_stage_task].p_end = current_cycle;
                pending_fetch_queue.push_back(busy_p_stage_task);
                busy_p_stage_task = -1;
            } else {
                // STALL: Hold resource
            }
        }
    }

    if (busy_p_stage_task == -1 && !pending_start_queue.empty()) {
        int task_id = pending_start_queue.front();
        pending_start_queue.pop_front();
        busy_p_stage_task = task_id;
        p_active = true;
        
        tasks[task_id].current_stage = Stage::PREDICT_HBM;
        tasks[task_id].p_start = current_cycle;
        //p_stage_free_cycle = current_cycle + calculate_p_latency(tasks[task_id]);
        if (per_group_p_latency == 0) {
            per_group_p_latency = calculate_p_latency_realistic(tasks[task_id]);
            std::cout << "Per-Group P-Stage Latency (Realistic): " << per_group_p_latency << " cycles\n";
        }
        p_stage_free_cycle = current_cycle + per_group_p_latency;
    }
    
    // --- Bubble Counting ---
    // A bubble is when ALL stages are idle (no active task processing)
    // Note: Stalled stages count as active (they are holding a resource)
    if (!p_active && !f_active && !c_active) {
        total_bubbles++;
    }
}

void PipelineSimulator::run() {
    bool all_finished = false;
    while (!all_finished) {
        current_cycle++;
        step();
        
        
        // Check termination
        all_finished = true;
        for (const auto& t : tasks) {
            if (t.current_stage != Stage::FINISHED) {
                all_finished = false;
                break;
            }
        }
        
        // Safety Break
        if (current_cycle > 10000000) break;
    }
}

const std::vector<HeadGroupTask>& PipelineSimulator::get_task_logs() const {
    return tasks;
}

} // namespace System
