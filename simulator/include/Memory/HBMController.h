#pragma once
#include "IMemoryController.h"
#include <deque>

namespace Memory {

class HBMController : public IMemoryController {
public:
    struct Config {
        double bandwidth_GBps;  // e.g., 400.0
        double frequency_GHz;   // e.g., 1.0 (to convert bytes/cycle)
        uint32_t latency_cycles;// Fixed pipeline latency
    };

    HBMController(Config config);

    bool send_request(const MemoryRequest& req) override;
    void step(size_t current_cycle) override;
    std::vector<MemoryRequest> pop_completed_requests() override;
    bool is_idle() const override;

private:
    Config config;
    double bytes_per_cycle;
    
    // Time when the HBM data bus will be free
    size_t bus_next_free_cycle = 0;

    // Requests being processed
    std::deque<MemoryRequest> pending_requests;
    
    // Requests finished this cycle
    std::vector<MemoryRequest> completed_buffer;
};

} // namespace Memory
