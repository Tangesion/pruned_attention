#include "Memory/DDRController.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace Memory {

DDRController::DDRController(Config config) : config(config) {
    bank_states.resize(config.num_channels);
    for (auto& channel_banks : bank_states) {
        channel_banks.resize(config.num_banks_per_channel);
    }
    channel_next_data_free_cycle.assign(config.num_channels, 0);

    stats.channel_accesses.assign(config.num_channels, 0);
    stats.channel_busy_cycles.assign(config.num_channels, 0);
    stats.bank_accesses.assign(
        static_cast<size_t>(config.num_channels) * config.num_banks_per_channel, 0);
    stats.bank_busy_cycles.assign(
        static_cast<size_t>(config.num_channels) * config.num_banks_per_channel, 0);
}

DDRController::PhysAddr DDRController::map_address(size_t address) const {
    PhysAddr phys;

    uint32_t channel_bits = static_cast<uint32_t>(std::log2(config.num_channels));
    uint32_t bank_bits = static_cast<uint32_t>(std::log2(config.num_banks_per_channel));
    uint32_t block_offset_bits = 6; // 64 bytes

    size_t blocks_per_row = std::max<size_t>(1, config.row_size_bytes >> block_offset_bits);
    uint32_t col_bits = static_cast<uint32_t>(std::log2(blocks_per_row));
    size_t block_addr = address >> block_offset_bits;
    uint32_t linear_channel = static_cast<uint32_t>(block_addr & (config.num_channels - 1));
    uint32_t linear_bank = static_cast<uint32_t>(
        (block_addr >> channel_bits) & (config.num_banks_per_channel - 1));
    size_t row = block_addr >> (channel_bits + bank_bits + col_bits);
    size_t col = (block_addr >> (channel_bits + bank_bits)) & (blocks_per_row - 1);

    if (config.mapping_strategy == BankMappingStrategy::LINEAR) {
        phys.channel = linear_channel;
        phys.bank = linear_bank;
    } else {
        size_t row_mix = row ^ (row >> channel_bits) ^ (row >> (channel_bits + bank_bits));
        uint32_t row_channel_hash =
            static_cast<uint32_t>(row_mix) & (config.num_channels - 1);
        uint32_t row_bank_hash = static_cast<uint32_t>(row_mix >> channel_bits) &
                                 (config.num_banks_per_channel - 1);

        // Permute channel/bank selection using row bits so power-of-two stream
        // offsets spread out, while accesses within the same row still preserve
        // their locality characteristics.
        phys.channel = (linear_channel ^ row_channel_hash) & (config.num_channels - 1);
        phys.bank = (linear_bank ^ row_bank_hash ^ row_channel_hash) &
                    (config.num_banks_per_channel - 1);
    }

    phys.row = row;
    phys.col = col;

    return phys;
}

bool DDRController::send_request(const MemoryRequest& req) {
    MemoryRequest processing_req = req;
    constexpr size_t kCacheLineBytes = 64;

    size_t total_bursts = std::max<size_t>(
        1, (req.size_bytes + kCacheLineBytes - 1) / kCacheLineBytes);
    size_t request_ready_cycle = req.arrival_cycle;

    stats.accepted_requests++;
    stats.total_bytes += req.size_bytes;
    stats.total_bursts += total_bursts;
    if (total_bursts > 1) {
        stats.multi_burst_requests++;
    }

    for (size_t burst_idx = 0; burst_idx < total_bursts; ++burst_idx) {
        size_t burst_address = req.address + burst_idx * kCacheLineBytes;
        PhysAddr phys = map_address(burst_address);

        if (phys.channel >= bank_states.size() || phys.bank >= bank_states[phys.channel].size()) {
            return false;
        }

        BankState& state = bank_states[phys.channel][phys.bank];
        size_t command_start_cycle = std::max(req.arrival_cycle, state.bank_next_free_cycle);
        size_t bank_ready_cycle = command_start_cycle;

        if (state.open_row_id == -1) {
            bank_ready_cycle += config.tRCD + config.tCL;
            state.open_row_id = static_cast<int64_t>(phys.row);
            stats.row_misses++;
        } else if (state.open_row_id == static_cast<int64_t>(phys.row)) {
            bank_ready_cycle += config.tCL;
            stats.row_hits++;
        } else {
            bank_ready_cycle += config.tRP + config.tRCD + config.tCL;
            state.open_row_id = static_cast<int64_t>(phys.row);
            stats.row_conflicts++;
        }

        size_t data_start_cycle = std::max(bank_ready_cycle, channel_next_data_free_cycle[phys.channel]);
        size_t data_end_cycle = data_start_cycle + config.tBURST;

        state.bank_next_free_cycle = data_end_cycle;
        channel_next_data_free_cycle[phys.channel] = data_end_cycle;
        request_ready_cycle = std::max(request_ready_cycle, data_end_cycle);

        ++stats.channel_accesses[phys.channel];
        stats.channel_busy_cycles[phys.channel] += data_end_cycle - data_start_cycle;

        size_t bank_index = flat_bank_index(phys.channel, phys.bank);
        ++stats.bank_accesses[bank_index];
        stats.bank_busy_cycles[bank_index] += data_end_cycle - command_start_cycle;
    }

    processing_req.ready_cycle = request_ready_cycle;
    pending_requests.push_back(processing_req);
    return true;
}

void DDRController::step(size_t current_cycle) {
    completed_buffer.clear();
    
    auto it = pending_requests.begin();
    while (it != pending_requests.end()) {
        if (it->ready_cycle <= current_cycle) {
            completed_buffer.push_back(*it);
            it = pending_requests.erase(it);
        } else {
            ++it;
        }
    }
}

std::vector<MemoryRequest> DDRController::pop_completed_requests() {
    return std::move(completed_buffer);
}

bool DDRController::is_idle() const {
    return pending_requests.empty() && completed_buffer.empty();
}

const DDRController::Stats& DDRController::get_stats() const {
    return stats;
}

const DDRController::Config& DDRController::get_config() const {
    return config;
}

size_t DDRController::flat_bank_index(uint32_t channel, uint32_t bank) const {
    return static_cast<size_t>(channel) * config.num_banks_per_channel + bank;
}

} // namespace Memory
