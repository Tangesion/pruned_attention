#pragma once

#include <cstdint>
namespace int4 {

inline int32_t sign_extend_int4(uint16_t val) {
    uint8_t v = val & 0xF;
    if (v & 0x8) {
        return static_cast<int32_t>(v) | 0xFFFFFFF0;
    } else {
        return static_cast<int32_t>(v);
    }
}
}