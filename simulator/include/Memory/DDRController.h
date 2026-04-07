#pragma once
#include "IMemoryController.h"
#include <vector>
#include <deque>
#include <map>

namespace Memory {

class DDRController : public IMemoryController {
public:
    struct Config {
        // Timing Parameters (in cycles)
        uint32_t tCL;   // Column Access Strobe Latency
        uint32_t tRCD;  // Row Address to Column Address Delay
        uint32_t tRP;   // Row Precharge Time
        uint32_t tBURST;// Data Burst Duration
        
        // Architecture
        uint32_t num_channels;
        uint32_t num_banks_per_channel;
        uint32_t row_size_bytes;
        
        // Mapping
        BankMappingStrategy mapping_strategy;
    };

    struct Stats {
        size_t accepted_requests = 0;
        size_t total_bytes = 0;
        size_t total_bursts = 0;
        size_t multi_burst_requests = 0;
        size_t row_hits = 0;
        size_t row_conflicts = 0;
        size_t row_misses = 0;
        std::vector<size_t> channel_accesses;
        std::vector<size_t> channel_busy_cycles;
        std::vector<size_t> bank_accesses;
        std::vector<size_t> bank_busy_cycles;
    };

    DDRController(Config config);

    bool send_request(const MemoryRequest& req) override;
    void step(size_t current_cycle) override;
    std::vector<MemoryRequest> pop_completed_requests() override;
    bool is_idle() const override;
    const Stats& get_stats() const;
    const Config& get_config() const;

private:
    Config config;

    struct BankState {
        int64_t open_row_id = -1;       // -1 indicates closed
        size_t bank_next_free_cycle = 0; // When can this bank accept a new command
    };

    // Organized as [Channel][Bank]
    std::vector<std::vector<BankState>> bank_states;
    std::vector<size_t> channel_next_data_free_cycle;

    // Requests waiting for simulation time to pass
    std::deque<MemoryRequest> pending_requests;
    std::vector<MemoryRequest> completed_buffer;
    Stats stats;

    // Helper to map address to physical location
    struct PhysAddr {
        uint32_t channel;
        uint32_t bank;
        size_t row;
        size_t col;
    };
    PhysAddr map_address(size_t address) const;
    size_t flat_bank_index(uint32_t channel, uint32_t bank) const;
};

} // namespace Memory
