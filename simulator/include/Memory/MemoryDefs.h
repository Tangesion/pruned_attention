#pragma once
#include <cstdint>
#include <cstddef>

namespace Memory {

// Memory Request Type
enum class RequestType {
    READ,
    WRITE
};

// A single memory request
struct MemoryRequest {
    size_t req_id;        // Unique ID for tracking
    size_t address;       // Logical Byte Address
    size_t size_bytes;      // Size of request
    RequestType type;
    
    size_t arrival_cycle; // When the request was sent to controller
    size_t ready_cycle;   // When the data is available (calculated by controller)

    MemoryRequest(size_t id, size_t addr, size_t size, RequestType t, size_t arrival)
        : req_id(id), address(addr), size_bytes(size), type(t), 
          arrival_cycle(arrival), ready_cycle(0) {}
};

// Strategy for DDR Bank Mapping
enum class BankMappingStrategy {
    LINEAR,     // Consecutive addresses -> Same Bank (Prone to conflicts)
    HASH_XOR    // XOR-based mapping (Reduces conflicts)
};

} // namespace Memory
