#include "Memory/HBMController.h"
#include <algorithm>
#include <cmath>

namespace Memory {

HBMController::HBMController(Config config) : config(config) {
    // Calculate bytes per cycle
    // GB/s / GHz = Bytes/ns / (Cycles/ns) = Bytes / Cycle
    this->bytes_per_cycle = config.bandwidth_GBps / config.frequency_GHz;
    this->bus_next_free_cycle = 0;
}

bool HBMController::send_request(const MemoryRequest& req) {
    // HBM usually has deep queues, for simulation we assume infinite queue for now
    // or we could limit pending_requests.size()
    
    MemoryRequest processing_req = req;
    
    // 1. Serialization Latency (Burst Time)
    size_t burst_cycles = static_cast<size_t>(std::ceil(req.size_bytes / bytes_per_cycle));
    
    // 2. Schedule the bus usage
    // The request can start only when the bus is free AND after it arrives
    size_t start_cycle = std::max(req.arrival_cycle, bus_next_free_cycle);
    
    // 3. Calculate Finish Time
    // Finish = Start + Fixed_Pipeline_Latency + Burst_Time
    processing_req.ready_cycle = start_cycle + config.latency_cycles + burst_cycles;
    
    // 4. Update Bus State
    // The bus is effectively busy transferring data during [start + latency, start + latency + burst]
    // Simplified: Next request can start pipelining such that data follows immediately.
    // We update 'bus_next_free_cycle' to represent when the *port* is ready for the *next* burst.
    // Note: This is a simplified "Leaky Bucket" model.
    bus_next_free_cycle = start_cycle + burst_cycles;

    pending_requests.push_back(processing_req);
    return true;
}

void HBMController::step(size_t current_cycle) {
    completed_buffer.clear();
    
    auto it = pending_requests.begin();
    while (it != pending_requests.end()) {
        if (it->ready_cycle <= current_cycle) {
            completed_buffer.push_back(*it);
            it = pending_requests.erase(it);
        } else {
            // Since we push in order of arrival, and ready_cycle is roughly monotonic
            // (modulo small variations if we had priorities, which we don't),
            // we can check others, but usually sticking to FIFO.
            ++it;
        }
    }
}

std::vector<MemoryRequest> HBMController::pop_completed_requests() {
    return std::move(completed_buffer);
}

bool HBMController::is_idle() const {
    return pending_requests.empty() && completed_buffer.empty();
}

} // namespace Memory
