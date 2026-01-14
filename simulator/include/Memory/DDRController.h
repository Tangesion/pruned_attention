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

    DDRController(Config config);

    bool send_request(const MemoryRequest& req) override;
    void step(uint64_t current_cycle) override;
    std::vector<MemoryRequest> pop_completed_requests() override;
    bool is_idle() const override;

private:
    Config config;

    struct BankState {
        int64_t open_row_id = -1;       // -1 indicates closed
        uint64_t bank_next_free_cycle = 0; // When can this bank accept a new command
    };

    // Organized as [Channel][Bank]
    std::vector<std::vector<BankState>> bank_states;

    // Requests waiting for simulation time to pass
    std::deque<MemoryRequest> pending_requests;
    std::vector<MemoryRequest> completed_buffer;

    // Helper to map address to physical location
    struct PhysAddr {
        uint32_t channel;
        uint32_t bank;
        uint64_t row;
        uint64_t col;
    };
    PhysAddr map_address(uint64_t address) const;
};

} // namespace Memory
