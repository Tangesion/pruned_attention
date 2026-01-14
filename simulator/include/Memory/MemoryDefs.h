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
    uint64_t req_id;        // Unique ID for tracking
    uint64_t address;       // Logical Byte Address
    size_t size_bytes;      // Size of request
    RequestType type;
    
    uint64_t arrival_cycle; // When the request was sent to controller
    uint64_t ready_cycle;   // When the data is available (calculated by controller)

    MemoryRequest(uint64_t id, uint64_t addr, size_t size, RequestType t, uint64_t arrival)
        : req_id(id), address(addr), size_bytes(size), type(t), 
          arrival_cycle(arrival), ready_cycle(0) {}
};

// Strategy for DDR Bank Mapping
enum class BankMappingStrategy {
    LINEAR,     // Consecutive addresses -> Same Bank (Prone to conflicts)
    HASH_XOR    // XOR-based mapping (Reduces conflicts)
};

} // namespace Memory
