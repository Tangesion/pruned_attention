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
}

DDRController::PhysAddr DDRController::map_address(uint64_t address) const {
    PhysAddr phys;
    
    // Simplified Mapping Logic
    // Physical Address breakdown usually: [Row | Bank | Channel | Col] or [Row | Col | Bank | Channel]
    // The paper focuses on Bank Mapping.
    
    // 1. Column Offset (Lower bits)
    // For simplicity, assume data is cache-line aligned (e.g., 64 bytes)
    // The 'col' bits are usually the lowest.
    // However, the paper implies we map "Logical Address" to "Bank".
    
    // Let's assume logical address is linear byte address.
    
    uint32_t channel_bits = static_cast<uint32_t>(std::log2(config.num_channels));
    uint32_t bank_bits    = static_cast<uint32_t>(std::log2(config.num_banks_per_channel));
    uint32_t row_size_bits= static_cast<uint32_t>(std::log2(config.row_size_bytes));
    
    // Basic Interleaving Granularity (e.g., 64 bytes)
    uint32_t block_offset_bits = 6; // 64 bytes
    
    uint64_t block_addr = address >> block_offset_bits;
    
    // Channel is usually the lowest interleaving
    phys.channel = block_addr & (config.num_channels - 1);
    uint64_t rest_addr = block_addr >> channel_bits;

    if (config.mapping_strategy == BankMappingStrategy::LINEAR) {
        // Linear: Consecutive blocks go to consecutive banks
        phys.bank = rest_addr & (config.num_banks_per_channel - 1);
        
        // Row is the rest
        // We need to account for how many blocks fit in a row.
        // RowSize = 2^row_size_bits.
        // Blocks per row = RowSize / 64.
        uint64_t blocks_per_row = config.row_size_bytes >> block_offset_bits;
        phys.row = rest_addr / (config.num_banks_per_channel * blocks_per_row);
        
    } else {
        // HASH_XOR Strategy
        // This is a simplified "Permutation-based" XOR mapping to randomize bank index.
        // bank = (lower_bits) ^ (higher_bits)
        
        uint32_t lower = rest_addr & (config.num_banks_per_channel - 1);
        uint32_t higher = (rest_addr >> bank_bits) & (config.num_banks_per_channel - 1);
        
        phys.bank = lower ^ higher;
        
        // Simplified Row calculation for simulation
        uint64_t blocks_per_row = config.row_size_bytes >> block_offset_bits;
        phys.row = rest_addr / (config.num_banks_per_channel * blocks_per_row);
    }
    
    return phys;
}

bool DDRController::send_request(const MemoryRequest& req) {
    MemoryRequest processing_req = req;
    
    // 1. Map Address
    PhysAddr phys = map_address(req.address);
    
    // 2. Get Bank State
    if (phys.channel >= bank_states.size() || phys.bank >= bank_states[0].size()) {
        // Should not happen if config is correct
        return false;
    }
    
    BankState& state = bank_states[phys.channel][phys.bank];
    
    // 3. Determine Start Cycle
    // The bank can start processing this command only after it's free from previous commands
    // AND after the request has arrived.
    uint64_t start_cycle = std::max(req.arrival_cycle, state.bank_next_free_cycle);
    uint64_t latency = 0;
    
    // 4. Calculate Latency based on Row State
    if (state.open_row_id == -1) {
        // Case: Row Closed (Page Empty)
        // Command: ACT + READ
        latency = config.tRCD + config.tCL + config.tBURST;
        state.open_row_id = phys.row;
    } else if (state.open_row_id == (int64_t)phys.row) {
        // Case: Row Hit (Page Open)
        // Command: READ
        latency = config.tCL + config.tBURST;
    } else {
        // Case: Row Conflict (Page Miss)
        // Command: PRE + ACT + READ
        latency = config.tRP + config.tRCD + config.tCL + config.tBURST;
        state.open_row_id = phys.row;
    }
    
    // 5. Update Request and Bank State
    processing_req.ready_cycle = start_cycle + latency;
    
    // Ideally, tRC (Row Cycle Time) defines when the bank is ready for a *different* row.
    // For simplicity, we just say the bank is busy until the data burst is done (plus some recovery).
    // In a detailed DDR model, we'd track tRAS, tRC, etc. strictly.
    // Here, we ensure the 'bank' (or at least the data path) is busy.
    state.bank_next_free_cycle = processing_req.ready_cycle;

    pending_requests.push_back(processing_req);
    return true;
}

void DDRController::step(uint64_t current_cycle) {
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

} // namespace Memory
