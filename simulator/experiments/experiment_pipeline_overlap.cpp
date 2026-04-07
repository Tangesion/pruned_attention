#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Memory/DDRController.h"
#include "System/PipelineSimulator.h"

using namespace Memory;
using namespace System;

namespace {

struct StageDurations {
    size_t p_cycles = 0;
    size_t f_cycles = 0;
    size_t c_cycles = 0;
};

struct WorkloadSpec {
    size_t context_length = 0;
    double sparsity_ratio = 0.0;
};

struct ExperimentResult {
    size_t context_length = 0;
    double sparsity_ratio = 0.0;
    int num_channels = 0;
    size_t total_cycles = 0;
    size_t serial_cycles = 0;
    double speedup_vs_serial = 0.0;
    double normalized_performance = 0.0;
    double avg_p_cycles = 0.0;
    double avg_f_cycles = 0.0;
    double avg_c_cycles = 0.0;
    size_t selected_kv_per_group = 0;
    std::vector<HeadGroupTask> logs;
    std::vector<StageDurations> durations;
};

constexpr int kFixedDdrChannels = 16;
constexpr size_t kRepresentativeContext = 64 * 1024;
constexpr double kRepresentativeSparsity = 0.10;

DDRController::Config make_ddr_config() {
    DDRController::Config ddr_cfg;
    ddr_cfg.tCL = 14;
    ddr_cfg.tRCD = 14;
    ddr_cfg.tRP = 14;
    ddr_cfg.tBURST = 4;
    ddr_cfg.num_channels = kFixedDdrChannels;
    ddr_cfg.num_banks_per_channel = 16;
    ddr_cfg.row_size_bytes = 8192;
    ddr_cfg.mapping_strategy = BankMappingStrategy::HASH_XOR;
    return ddr_cfg;
}

PipelineConfig make_pipeline_config(const WorkloadSpec& workload) {
    PipelineConfig pipe_cfg;
    pipe_cfg.frequency_mhz = 200.0;
    pipe_cfg.max_queue_size = 8;
    pipe_cfg.verbose_logging = false;
    pipe_cfg.use_realistic_stage_models = false;
    pipe_cfg.use_hash_aware_fetch = true;
    pipe_cfg.use_dual_fetch = true;
    pipe_cfg.fetch_num_lanes = 2;
    pipe_cfg.max_sgu_merge_bytes = 16 * 1024;
    pipe_cfg.fetch_max_inflight_bursts = 1024;
    pipe_cfg.fetch_head_stride_padding_bytes = 64 * kFixedDdrChannels * 16;

    pipe_cfg.context_length = workload.context_length;
    pipe_cfg.head_dim = 128;
    pipe_cfg.bytes_per_elem = 2;
    pipe_cfg.sparsity_ratio = workload.sparsity_ratio;
    pipe_cfg.num_head_groups = 8;
    pipe_cfg.heads_per_group = 4;
    pipe_cfg.selected_kv_len =
        static_cast<size_t>(pipe_cfg.context_length * pipe_cfg.sparsity_ratio);

    pipe_cfg.hbm_scan_bandwidth_gbps = 400.0;
    pipe_cfg.int4_vector_size_bytes = 64;
    pipe_cfg.P_stage_pe_nums = 2048;
    pipe_cfg.num_bf16_pes = 512;
    pipe_cfg.C_stage_pe_nums = 256;

    pipe_cfg.uram_bandwidth_gbps = 1000.0;
    pipe_cfg.uram_latency_cycles = 5;
    pipe_cfg.tfu_throughput_per_cycle = 64;
    return pipe_cfg;
}

size_t compute_serial_cycles(const std::vector<StageDurations>& durations) {
    size_t total = 0;
    for (const auto& item : durations) {
        total += item.p_cycles + item.f_cycles + item.c_cycles;
    }
    return total;
}

ExperimentResult run_workload(const WorkloadSpec& workload) {
    DDRController ddr(make_ddr_config());
    PipelineConfig pipe_cfg = make_pipeline_config(workload);
    PipelineSimulator sim(pipe_cfg, ddr);
    sim.run();

    ExperimentResult result;
    result.context_length = workload.context_length;
    result.sparsity_ratio = workload.sparsity_ratio;
    result.num_channels = kFixedDdrChannels;
    result.total_cycles = sim.get_total_cycles();
    result.logs = sim.get_task_logs();
    result.durations.reserve(result.logs.size());

    size_t total_p_cycles = 0;
    size_t total_f_cycles = 0;
    size_t total_c_cycles = 0;

    for (const auto& task : result.logs) {
        StageDurations durations;
        durations.p_cycles = task.p_end - task.p_start;
        durations.f_cycles = task.f_end - task.f_start;
        durations.c_cycles = task.c_end - task.c_start;
        result.durations.push_back(durations);

        total_p_cycles += durations.p_cycles;
        total_f_cycles += durations.f_cycles;
        total_c_cycles += durations.c_cycles;
        result.selected_kv_per_group = task.selected_kv_len;
    }

    result.serial_cycles = compute_serial_cycles(result.durations);
    result.speedup_vs_serial =
        static_cast<double>(result.serial_cycles) / static_cast<double>(result.total_cycles);
    result.normalized_performance = result.speedup_vs_serial;

    const double group_count = static_cast<double>(result.logs.size());
    result.avg_p_cycles = total_p_cycles / group_count;
    result.avg_f_cycles = total_f_cycles / group_count;
    result.avg_c_cycles = total_c_cycles / group_count;
    return result;
}

std::string format_context_label(size_t context_length) {
    return std::to_string(context_length / 1024) + "K";
}

std::string format_percent(double ratio) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(0) << ratio * 100.0 << "%";
    return oss.str();
}

void write_summary_csv(
    const std::filesystem::path& csv_path,
    const std::vector<ExperimentResult>& results) {
    std::ofstream out(csv_path);
    out << "context_length,sparsity_ratio,num_channels,selected_kv_per_group,total_cycles,"
           "serial_cycles,speedup_vs_serial,normalized_performance,avg_p_cycles,avg_f_cycles,avg_c_cycles\n";
    out << std::fixed << std::setprecision(4);
    for (const auto& result : results) {
        out << result.context_length << ','
            << result.sparsity_ratio << ','
            << result.num_channels << ','
            << result.selected_kv_per_group << ','
            << result.total_cycles << ','
            << result.serial_cycles << ','
            << result.speedup_vs_serial << ','
            << result.normalized_performance << ','
            << result.avg_p_cycles << ','
            << result.avg_f_cycles << ','
            << result.avg_c_cycles << '\n';
    }
}

void write_timeline_csv(
    const std::filesystem::path& csv_path,
    const ExperimentResult& representative_result) {
    std::ofstream out(csv_path);
    out << "context_length,sparsity_ratio,num_channels,group_id,stage,start_cycle,end_cycle,duration_cycles\n";
    for (const auto& task : representative_result.logs) {
        out << representative_result.context_length << ','
            << representative_result.sparsity_ratio << ','
            << representative_result.num_channels << ','
            << task.group_id << ",P,"
            << task.p_start << ','
            << task.p_end << ','
            << (task.p_end - task.p_start) << '\n';
        out << representative_result.context_length << ','
            << representative_result.sparsity_ratio << ','
            << representative_result.num_channels << ','
            << task.group_id << ",F,"
            << task.f_start << ','
            << task.f_end << ','
            << (task.f_end - task.f_start) << '\n';
        out << representative_result.context_length << ','
            << representative_result.sparsity_ratio << ','
            << representative_result.num_channels << ','
            << task.group_id << ",C,"
            << task.c_start << ','
            << task.c_end << ','
            << (task.c_end - task.c_start) << '\n';
    }
}

void print_summary_table(const std::vector<ExperimentResult>& results) {
    std::cout << "=================================================================\n";
    std::cout << " Experiment 3: Pipeline Overlap Across Workloads\n";
    std::cout << "=================================================================\n";
    std::cout << "Hardware fixed at 16 DDR channels | Head Groups: 8 x 4 | Metric: serial / pipelined latency\n";
    std::cout << std::left
              << std::setw(10) << "Context"
              << std::setw(10) << "Sparsity"
              << std::setw(12) << "Sel/Grp"
              << std::setw(14) << "Pipeline"
              << std::setw(14) << "Serial"
              << std::setw(10) << "Speedup"
              << std::setw(12) << "Avg P"
              << std::setw(12) << "Avg F"
              << "Avg C\n";
    std::cout << "---------------------------------------------------------------------------------------------\n";

    for (const auto& result : results) {
        std::cout << std::left
                  << std::setw(10) << format_context_label(result.context_length)
                  << std::setw(10) << format_percent(result.sparsity_ratio)
                  << std::setw(12) << result.selected_kv_per_group
                  << std::setw(14) << result.total_cycles
                  << std::setw(14) << result.serial_cycles
                  << std::setw(10) << std::fixed << std::setprecision(2)
                  << result.speedup_vs_serial
                  << std::setw(12) << std::fixed << std::setprecision(0)
                  << result.avg_p_cycles
                  << std::setw(12) << std::fixed << std::setprecision(0)
                  << result.avg_f_cycles
                  << std::fixed << std::setprecision(0) << result.avg_c_cycles << '\n';
    }
    std::cout << "---------------------------------------------------------------------------------------------\n";
    std::cout << "Representative Gantt workload: "
              << format_context_label(kRepresentativeContext) << ", "
              << format_percent(kRepresentativeSparsity) << " sparsity\n";
}

} // namespace

int main() {
    const std::vector<size_t> context_lengths = {
        16 * 1024,
        32 * 1024,
        64 * 1024,
        128 * 1024,
        256 * 1024,
        512 * 1024,
    };
    const std::vector<double> sparsity_ratios = {
        0.01,
        0.05,
        0.10,
        0.20,
    };

    std::vector<ExperimentResult> results;
    results.reserve(context_lengths.size() * sparsity_ratios.size());

    ExperimentResult representative_result;
    bool found_representative = false;

    for (size_t context_length : context_lengths) {
        for (double sparsity_ratio : sparsity_ratios) {
            ExperimentResult result = run_workload({context_length, sparsity_ratio});
            if (context_length == kRepresentativeContext &&
                std::abs(sparsity_ratio - kRepresentativeSparsity) < 1e-9) {
                representative_result = result;
                found_representative = true;
            }
            results.push_back(std::move(result));
        }
    }

    if (!found_representative) {
        std::cerr << "Representative workload result was not generated.\n";
        return 1;
    }

    print_summary_table(results);

    const std::filesystem::path results_dir =
        std::filesystem::path("simulator") / "results";
    std::filesystem::create_directories(results_dir);

    const std::filesystem::path summary_csv =
        results_dir / "pipeline_overlap_workload_summary.csv";
    const std::filesystem::path timeline_csv =
        results_dir / "pipeline_overlap_representative_timeline.csv";
    write_summary_csv(summary_csv, results);
    write_timeline_csv(timeline_csv, representative_result);

    std::cout << "Summary CSV written to: " << summary_csv << '\n';
    std::cout << "Timeline CSV written to: " << timeline_csv << '\n';
    return 0;
}
