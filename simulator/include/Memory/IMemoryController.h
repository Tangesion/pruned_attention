#pragma once
#include "MemoryDefs.h"
#include <deque>
#include <vector>
#include <optional>

namespace Memory {

class IMemoryController {
public:
    virtual ~IMemoryController() = default;

    // Send a request to the memory subsystem
    // Returns true if accepted, false if buffer is full (backpressure)
    virtual bool send_request(const MemoryRequest& req) = 0;

    // Advance simulation by one cycle
    virtual void step(size_t current_cycle) = 0;

    // Get requests that completed in the *current* cycle
    // The user should call this after step()
    virtual std::vector<MemoryRequest> pop_completed_requests() = 0;
    
    // Check if the memory system is idle (no pending requests)
    virtual bool is_idle() const = 0;
};

} // namespace Memory
