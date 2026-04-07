#include "System/PipelineSimulator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace System {

namespace {
size_t per_group_p_latency = 0;
size_t per_group_c_latency = 0;

constexpr size_t kCacheLineBytes = 64;
constexpr size_t kLargeContextThreshold = 256 * 1024;
constexpr size_t kHashBucketChunk = 8;
constexpr size_t kLargeHashBucketChunk = 4096;

size_t ceil_div(size_t numerator, size_t denominator) {
    return (numerator + denominator - 1) / denominator;
}

struct RequestPlacement {
    uint32_t channel = 0;
    uint32_t bank = 0;
};

std::vector<size_t> generate_sparse_indices(size_t count, size_t context_length, uint32_t seed) {
    count = std::min(count, context_length);
    if (count == 0 || context_length == 0) {
        return {};
    }

    std::mt19937 gen(seed);
    std::vector<uint8_t> used(context_length, 0);
    std::vector<size_t> indices;
    indices.reserve(count);

    const size_t min_cluster_len = 8;
    const size_t max_cluster_len = 128;
    const size_t base_cluster_len = std::min(
        max_cluster_len,
        std::max(
            min_cluster_len,
            static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(count))))));
    std::uniform_int_distribution<size_t> cluster_len_dist(min_cluster_len, base_cluster_len);

    while (indices.size() < count) {
        const size_t remaining = count - indices.size();
        const size_t cluster_len = std::min(remaining, cluster_len_dist(gen));
        const size_t max_start = context_length > cluster_len ? (context_length - cluster_len) : 0;
        std::uniform_int_distribution<size_t> start_dist(0, max_start);
        const size_t start = start_dist(gen);

        for (size_t offset = 0; offset < cluster_len && indices.size() < count; ++offset) {
            const size_t index = start + offset;
            if (index >= context_length || used[index]) {
                continue;
            }
            used[index] = 1;
            indices.push_back(index);
        }
    }

    std::sort(indices.begin(), indices.end());
    return indices;
}

std::vector<Memory::MemoryRequest> build_request_stream(
    const std::vector<size_t>& addresses,
    size_t token_kv_size,
    size_t max_merge_bytes) {
    std::vector<Memory::MemoryRequest> requests;
    requests.reserve(addresses.size());

    if (addresses.empty()) {
        return requests;
    }

    const size_t merge_limit = std::max(token_kv_size, max_merge_bytes);
    size_t next_request_id = 0;

    if (merge_limit <= token_kv_size) {
        for (size_t address : addresses) {
            requests.emplace_back(next_request_id++, address, token_kv_size, Memory::RequestType::READ, 0);
        }
        return requests;
    }

    size_t burst_start = addresses.front();
    size_t burst_size = token_kv_size;
    for (size_t index = 1; index < addresses.size(); ++index) {
        if (addresses[index] == addresses[index - 1] + token_kv_size &&
            burst_size + token_kv_size <= merge_limit) {
            burst_size += token_kv_size;
            continue;
        }

        requests.emplace_back(next_request_id++, burst_start, burst_size, Memory::RequestType::READ, 0);
        burst_start = addresses[index];
        burst_size = token_kv_size;
    }
    requests.emplace_back(next_request_id++, burst_start, burst_size, Memory::RequestType::READ, 0);
    return requests;
}

std::vector<Memory::MemoryRequest> interleave_request_streams(
    const std::vector<std::vector<Memory::MemoryRequest>>& request_streams) {
    size_t total_requests = 0;
    for (const auto& stream : request_streams) {
        total_requests += stream.size();
    }

    std::vector<Memory::MemoryRequest> merged;
    merged.reserve(total_requests);
    std::vector<size_t> offsets(request_streams.size(), 0);
    size_t next_request_id = 0;

    bool has_progress = true;
    while (has_progress) {
        has_progress = false;
        for (size_t stream_id = 0; stream_id < request_streams.size(); ++stream_id) {
            if (offsets[stream_id] >= request_streams[stream_id].size()) {
                continue;
            }

            const auto& request = request_streams[stream_id][offsets[stream_id]++];
            merged.emplace_back(
                next_request_id++,
                request.address,
                request.size_bytes,
                request.type,
                0);
            has_progress = true;
        }
    }

    return merged;
}

RequestPlacement estimate_request_placement(
    size_t address,
    const Memory::DDRController::Config& ddr_config) {
    RequestPlacement placement;

    const uint32_t channel_bits = static_cast<uint32_t>(std::log2(ddr_config.num_channels));
    const uint32_t bank_bits = static_cast<uint32_t>(std::log2(ddr_config.num_banks_per_channel));
    const size_t blocks_per_row = std::max<size_t>(1, ddr_config.row_size_bytes >> 6);
    const uint32_t col_bits = static_cast<uint32_t>(std::log2(blocks_per_row));
    const size_t block_addr = address >> 6;
    const size_t row = block_addr >> (channel_bits + bank_bits + col_bits);

    const uint32_t linear_channel =
        static_cast<uint32_t>(block_addr & (ddr_config.num_channels - 1));
    const uint32_t linear_bank = static_cast<uint32_t>(
        (block_addr >> channel_bits) & (ddr_config.num_banks_per_channel - 1));

    if (ddr_config.mapping_strategy == Memory::BankMappingStrategy::LINEAR) {
        placement.channel = linear_channel;
        placement.bank = linear_bank;
        return placement;
    }

    const size_t row_mix = row ^ (row >> channel_bits) ^ (row >> (channel_bits + bank_bits));
    const uint32_t row_channel_hash =
        static_cast<uint32_t>(row_mix) & (ddr_config.num_channels - 1);
    const uint32_t row_bank_hash = static_cast<uint32_t>(row_mix >> channel_bits) &
                                   (ddr_config.num_banks_per_channel - 1);

    placement.channel = (linear_channel ^ row_channel_hash) & (ddr_config.num_channels - 1);
    placement.bank = (linear_bank ^ row_bank_hash ^ row_channel_hash) &
                     (ddr_config.num_banks_per_channel - 1);
    return placement;
}

std::vector<Memory::MemoryRequest> rebalance_requests_by_channel_bank(
    const std::vector<Memory::MemoryRequest>& requests,
    const Memory::DDRController::Config& ddr_config,
    bool bank_only_buckets) {
    const size_t num_buckets = bank_only_buckets
                                   ? ddr_config.num_banks_per_channel
                                   : static_cast<size_t>(ddr_config.num_channels) *
                                         ddr_config.num_banks_per_channel;
    std::vector<std::vector<Memory::MemoryRequest>> buckets(num_buckets);
    for (const auto& request : requests) {
        const auto placement = estimate_request_placement(request.address, ddr_config);
        const size_t bucket_index = bank_only_buckets
                                        ? placement.bank
                                        : static_cast<size_t>(placement.channel) *
                                              ddr_config.num_banks_per_channel +
                                              placement.bank;
        buckets[bucket_index].push_back(request);
    }

    for (auto& bucket : buckets) {
        std::sort(bucket.begin(), bucket.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.address != rhs.address) {
                return lhs.address < rhs.address;
            }
            return lhs.size_bytes < rhs.size_bytes;
        });
    }

    std::vector<Memory::MemoryRequest> reordered;
    reordered.reserve(requests.size());
    std::vector<size_t> offsets(buckets.size(), 0);
    const size_t bucket_chunk = bank_only_buckets ? kLargeHashBucketChunk : kHashBucketChunk;
    size_t next_request_id = 0;

    bool has_progress = true;
    while (has_progress) {
        has_progress = false;
        for (size_t bucket = 0; bucket < buckets.size(); ++bucket) {
            for (size_t chunk = 0;
                 chunk < bucket_chunk && offsets[bucket] < buckets[bucket].size();
                 ++chunk) {
                const auto& request = buckets[bucket][offsets[bucket]++];
                reordered.emplace_back(
                    next_request_id++,
                    request.address,
                    request.size_bytes,
                    request.type,
                    0);
                has_progress = true;
            }
        }
    }

    return reordered;
}

size_t request_burst_count(const Memory::MemoryRequest& request) {
    return std::max<size_t>(1, ceil_div(request.size_bytes, kCacheLineBytes));
}

size_t run_ddr_simulation(
    const Memory::DDRController::Config& ddr_config,
    std::vector<Memory::MemoryRequest> workload,
    size_t max_in_flight_bursts) {
    if (workload.empty()) {
        return 1;
    }

    Memory::DDRController controller(ddr_config);
    size_t current_cycle = 0;
    size_t request_index = 0;
    size_t completed_count = 0;
    size_t in_flight_bursts = 0;

    while (completed_count < workload.size()) {
        while (request_index < workload.size()) {
            const size_t request_bursts = request_burst_count(workload[request_index]);
            if (in_flight_bursts + request_bursts > max_in_flight_bursts) {
                break;
            }

            workload[request_index].arrival_cycle = current_cycle;
            if (!controller.send_request(workload[request_index])) {
                break;
            }
            in_flight_bursts += request_bursts;
            ++request_index;
        }

        controller.step(current_cycle);
        auto completed = controller.pop_completed_requests();
        if (!completed.empty()) {
            completed_count += completed.size();
            for (const auto& request : completed) {
                in_flight_bursts -= request_burst_count(request);
            }
        }

        ++current_cycle;
        if (current_cycle > 100000000) {
            return current_cycle;
        }
    }

    return std::max<size_t>(1, current_cycle);
}

std::vector<Memory::MemoryRequest> build_lane_workload(
    const HeadGroupTask& task,
    const PipelineConfig& pipeline_config,
    const Memory::DDRController::Config& ddr_config,
    const std::vector<size_t>& shared_indices,
    size_t head_begin,
    size_t head_end) {
    const size_t token_kv_size = pipeline_config.head_dim * pipeline_config.bytes_per_elem * 2;
    const size_t bytes_per_head = task.seq_len * token_kv_size;
    const size_t rotation_padding = pipeline_config.fetch_head_stride_padding_bytes > 0
                                        ? pipeline_config.fetch_head_stride_padding_bytes
                                        : kCacheLineBytes * ddr_config.num_channels *
                                              ddr_config.num_banks_per_channel;
    const size_t padded_head_stride = bytes_per_head + rotation_padding;
    const size_t merge_bytes = pipeline_config.use_hash_aware_fetch
                                   ? std::max(token_kv_size, pipeline_config.max_sgu_merge_bytes)
                                   : token_kv_size;

    std::vector<std::vector<Memory::MemoryRequest>> request_streams;
    request_streams.reserve(head_end - head_begin);

    for (size_t head = head_begin; head < head_end; ++head) {
        std::vector<size_t> addresses;
        addresses.reserve(shared_indices.size());

        const size_t global_head_id = static_cast<size_t>(task.group_id) * task.num_heads + head;
        const size_t head_base = global_head_id * padded_head_stride;
        for (size_t token_index : shared_indices) {
            addresses.push_back(head_base + token_index * token_kv_size);
        }

        request_streams.push_back(build_request_stream(addresses, token_kv_size, merge_bytes));
    }

    auto workload = interleave_request_streams(request_streams);
    if (pipeline_config.use_hash_aware_fetch &&
        ddr_config.mapping_strategy == Memory::BankMappingStrategy::HASH_XOR) {
        const bool use_bank_only_buckets = task.seq_len >= kLargeContextThreshold;
        workload = rebalance_requests_by_channel_bank(workload, ddr_config, use_bank_only_buckets);
    }

    return workload;
}

} // namespace

PipelineSimulator::PipelineSimulator(PipelineConfig config, Memory::DDRController& ddr_controller)
    : config(config), ddr(ddr_controller),
      hbm(Memory::HBMController::Config{
          config.hbm_scan_bandwidth_gbps,
          config.frequency_mhz / 1000.0,
          20
      }) {
    per_group_p_latency = 0;
    per_group_c_latency = 0;

    for (int i = 0; i < config.num_head_groups; ++i) {
        HeadGroupTask task;
        task.group_id = i;
        task.num_heads = config.heads_per_group;
        task.seq_len = config.context_length;
        task.selected_kv_len = static_cast<size_t>(config.context_length * config.sparsity_ratio);
        task.P_stage_pe_nums = config.P_stage_pe_nums;
        task.C_stage_pe_nums = config.C_stage_pe_nums;
        task.current_stage = Stage::IDLE;
        tasks.push_back(task);
        pending_start_queue.push_back(i);
    }

    auto predict_mult_pipe_factory = []() { return std::make_unique<int4::Int4MultiplyPipeline>(1); };
    auto predict_add_pipe_factory = []() { return std::make_unique<int4::Int4AddPipeline>(1); };
    auto predict_mac_array = std::make_unique<PE::ReduceMacArrayUnit>(
        predict_mult_pipe_factory,
        predict_add_pipe_factory,
        config.P_stage_pe_nums);
    PE::GemvScheduler::Config predict_config{config.P_stage_pe_nums};
    predict_gemv_scheduler =
        std::make_unique<PE::GemvScheduler>(std::move(predict_mac_array), predict_config);

    auto compute_mult_pipe_factory = []() { return std::make_unique<bf16::BF16MultiplyPipeline>(); };
    auto compute_add_pipe_factory = []() { return std::make_unique<bf16::BF16AddPipeline>(); };
    auto compute_mac_array = std::make_unique<PE::ReduceMacArrayUnit>(
        compute_mult_pipe_factory,
        compute_add_pipe_factory,
        config.C_stage_pe_nums);
    PE::GemvScheduler::Config compute_config{config.C_stage_pe_nums};
    compute_gemv_scheduler =
        std::make_unique<PE::GemvScheduler>(std::move(compute_mac_array), compute_config);
}

size_t PipelineSimulator::calculate_p_latency(const HeadGroupTask& task) {
    const size_t total_scan_bytes = task.seq_len * task.num_heads * config.int4_vector_size_bytes;
    const double hbm_bytes_per_cycle = config.hbm_scan_bandwidth_gbps / (config.frequency_mhz / 1000.0);
    const size_t hbm_cycles = static_cast<size_t>(
        std::ceil(static_cast<double>(total_scan_bytes) / hbm_bytes_per_cycle));

    const size_t total_macs = task.seq_len * config.head_dim * task.num_heads;
    const size_t predict_ops_per_cycle = std::max<size_t>(1, task.P_stage_pe_nums);
    const size_t compute_cycles = static_cast<size_t>(
        std::ceil(static_cast<double>(total_macs) / predict_ops_per_cycle)) + 44;

    const size_t tfu_cycles = static_cast<size_t>(
        std::ceil(static_cast<double>(task.seq_len) / std::max<size_t>(1, config.tfu_throughput_per_cycle)));
    const size_t num_candidates = std::max<size_t>(1, task.selected_kv_len);
    const size_t candidate_bytes = num_candidates * 8;
    const double uram_bytes_per_cycle = config.uram_bandwidth_gbps / (config.frequency_mhz / 1000.0);
    const size_t uram_write_cycles = static_cast<size_t>(
        std::ceil(static_cast<double>(candidate_bytes) / uram_bytes_per_cycle));

    const size_t bottleneck_cycles =
        std::max({hbm_cycles, compute_cycles, tfu_cycles, uram_write_cycles});
    const size_t pipeline_overhead = 100;
    return std::max<size_t>(1, bottleneck_cycles + pipeline_overhead);
}

size_t PipelineSimulator::calculate_f_latency(const HeadGroupTask& task) {
    const auto& ddr_config = ddr.get_config();
    const size_t selected_tokens = std::max<size_t>(1, task.selected_kv_len);
    const auto shared_indices = generate_sparse_indices(
        selected_tokens,
        task.seq_len,
        2026u + static_cast<uint32_t>(task.group_id));

    const size_t lane_count = config.use_dual_fetch
                                  ? std::max<size_t>(1, std::min(config.fetch_num_lanes, static_cast<size_t>(task.num_heads)))
                                  : 1;
    const size_t heads_per_lane = ceil_div(static_cast<size_t>(task.num_heads), lane_count);

    size_t max_lane_cycles = 0;
    for (size_t lane = 0; lane < lane_count; ++lane) {
        const size_t head_begin = lane * heads_per_lane;
        const size_t head_end = std::min(static_cast<size_t>(task.num_heads), head_begin + heads_per_lane);
        if (head_begin >= head_end) {
            continue;
        }

        auto workload = build_lane_workload(
            task,
            config,
            ddr_config,
            shared_indices,
            head_begin,
            head_end);
        const size_t lane_cycles = run_ddr_simulation(
            ddr_config,
            std::move(workload),
            std::max<size_t>(64, config.fetch_max_inflight_bursts));
        max_lane_cycles = std::max(max_lane_cycles, lane_cycles);
    }

    if (lane_count > 1) {
        max_lane_cycles = static_cast<size_t>(std::ceil(static_cast<double>(max_lane_cycles) * 1.05));
    }

    return std::max<size_t>(1, max_lane_cycles);
}

size_t PipelineSimulator::calculate_c_latency(const HeadGroupTask& task) {
    const size_t kv_tokens = std::max<size_t>(1, task.selected_kv_len);
    const double total_macs = static_cast<double>(2ULL * kv_tokens * config.head_dim * task.num_heads);
    const size_t ops_per_cycle = std::max<size_t>(1, task.C_stage_pe_nums);
    const size_t cycles = static_cast<size_t>(std::ceil(total_macs / ops_per_cycle)) + 50;
    return std::max<size_t>(1, cycles);
}

size_t PipelineSimulator::calculate_p_latency_realistic(const HeadGroupTask& task) {
    size_t num_heads = task.num_heads;
    size_t seq_len = task.seq_len;
    size_t head_dim = config.head_dim;

    std::vector<std::vector<uint16_t>> q(num_heads);
    std::vector<std::vector<std::vector<uint16_t>>> k(num_heads);
    System::generate_random_int4_qk(num_heads, seq_len, head_dim, q, k);

    predict_gemv_scheduler->run_gemv_int32(k, q);
    size_t compute_cycles = predict_gemv_scheduler->get_total_cycles();

    size_t total_scan_bytes = seq_len * num_heads * config.int4_vector_size_bytes;
    double bytes_per_cycle = config.hbm_scan_bandwidth_gbps / (config.frequency_mhz / 1000.0);
    size_t hbm_cycles = static_cast<size_t>(std::ceil(total_scan_bytes / bytes_per_cycle));

    size_t tfu_cycles = seq_len / config.tfu_throughput_per_cycle;
    size_t num_candidates = static_cast<size_t>(seq_len * config.sparsity_ratio);
    size_t candidate_bytes = num_candidates * 8;

    double uram_bytes_per_cycle = config.uram_bandwidth_gbps / (config.frequency_mhz / 1000.0);
    size_t uram_write_cycles = static_cast<size_t>(std::ceil(candidate_bytes / uram_bytes_per_cycle));

    size_t bottleneck_cycles = std::max({hbm_cycles, compute_cycles, tfu_cycles, uram_write_cycles});
    size_t pipeline_overhead = 100;
    size_t total_cycles = bottleneck_cycles + pipeline_overhead;

    if (config.verbose_logging) {
        std::cout << "[P-Stage Detailed Breakdown]\n"
                  << "  - HBM Scan Bytes: " << total_scan_bytes << " -> " << hbm_cycles
                  << " cycles (BW Limit)\n"
                  << "  - Int4 GEMV Compute: " << compute_cycles << " cycles (ALU Limit)\n"
                  << "  - TFU Filtering: " << tfu_cycles << " cycles (Logic Limit)\n"
                  << "  - URAM Write (" << num_candidates << " cands): " << uram_write_cycles
                  << " cycles (BW Limit)\n"
                  << "  => Bottleneck: " << bottleneck_cycles << " cycles\n"
                  << "  => Total Latency: " << total_cycles << " cycles (Streaming)\n";
    }

    return total_cycles;
}

size_t PipelineSimulator::calculate_c_latency_realistic(const HeadGroupTask& task) {
    size_t num_heads = task.num_heads;
    size_t selected_kv_len = task.selected_kv_len;
    size_t head_dim = config.head_dim;

    std::vector<std::vector<uint16_t>> q(num_heads);
    std::vector<std::vector<std::vector<uint16_t>>> k(num_heads);
    std::vector<std::vector<std::vector<uint16_t>>> v(num_heads);

    System::generate_random_bf16_qkv(num_heads, selected_kv_len, head_dim, q, k, v);

    auto result1 = compute_gemv_scheduler->run_gemv(k, q);
    auto result2 = compute_gemv_scheduler->run_gemv(v, result1);
    (void)result2;
    return compute_gemv_scheduler->get_total_cycles();
}

void PipelineSimulator::step() {
    bool p_active = false;
    bool f_active = false;
    bool c_active = false;

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
        if (per_group_c_latency == 0) {
            per_group_c_latency = config.use_realistic_stage_models
                                      ? calculate_c_latency_realistic(tasks[task_id])
                                      : calculate_c_latency(tasks[task_id]);
            if (config.verbose_logging) {
                std::cout << "Per-Group C-Stage Latency: " << per_group_c_latency << " cycles\n";
            }
        }
        c_stage_free_cycle = current_cycle + per_group_c_latency;
    }

    if (busy_f_stage_task != -1) {
        f_active = true;
        if (current_cycle >= f_stage_free_cycle) {
            if (pending_compute_queue.size() < config.max_queue_size) {
                tasks[busy_f_stage_task].f_end = current_cycle;
                pending_compute_queue.push_back(busy_f_stage_task);
                busy_f_stage_task = -1;
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
        const size_t f_latency = calculate_f_latency(tasks[task_id]);
        if (config.verbose_logging) {
            std::cout << "Group " << task_id << " F-Stage Latency: " << f_latency << " cycles\n";
        }
        f_stage_free_cycle = current_cycle + f_latency;
    }

    if (busy_p_stage_task != -1) {
        p_active = true;
        if (current_cycle >= p_stage_free_cycle) {
            if (pending_fetch_queue.size() < config.max_queue_size) {
                tasks[busy_p_stage_task].p_end = current_cycle;
                pending_fetch_queue.push_back(busy_p_stage_task);
                busy_p_stage_task = -1;
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
        if (per_group_p_latency == 0) {
            per_group_p_latency = config.use_realistic_stage_models
                                      ? calculate_p_latency_realistic(tasks[task_id])
                                      : calculate_p_latency(tasks[task_id]);
            if (config.verbose_logging) {
                std::cout << "Per-Group P-Stage Latency: " << per_group_p_latency << " cycles\n";
            }
        }
        p_stage_free_cycle = current_cycle + per_group_p_latency;
    }

    if (!p_active && !f_active && !c_active) {
        total_bubbles++;
    }
}

void PipelineSimulator::run() {
    bool all_finished = false;
    while (!all_finished) {
        current_cycle++;
        step();

        all_finished = true;
        for (const auto& task : tasks) {
            if (task.current_stage != Stage::FINISHED) {
                all_finished = false;
                break;
            }
        }

        if (current_cycle > 100000000) {
            break;
        }
    }
}

const std::vector<HeadGroupTask>& PipelineSimulator::get_task_logs() const {
    return tasks;
}

} // namespace System
