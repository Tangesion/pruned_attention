#pragma once
#include <cstdint>
#include <stdexcept>
#include <variant>

namespace PE {

enum class TypeCode : uint8_t {
    UInt16 = 0,
    Int32 = 1,
    Float = 2
};

struct DataType {
    TypeCode code;
    uint8_t bits;
    
    bool operator==(const DataType& other) const {
        return code == other.code && bits == other.bits;
    }
    bool operator!=(const DataType& other) const {
        return !(*this == other);
    }
};

struct Number {
    DataType type;
    union {
        uint16_t u16;
        int32_t i32;
        float f32;
    } value;

    Number() : type{TypeCode::UInt16, 16} { value.u16 = 0; }
    
    explicit Number(uint16_t v) : type{TypeCode::UInt16, 16} { value.u16 = v; }
    explicit Number(int32_t v) : type{TypeCode::Int32, 32} { value.i32 = v; }
    explicit Number(float v) : type{TypeCode::Float, 32} { value.f32 = v; }
    Number(DataType t, size_t raw_val) : type(t) {
        if (t.code == TypeCode::UInt16) value.u16 = (uint16_t)raw_val;
        else if (t.code == TypeCode::Int32) value.i32 = (int32_t)raw_val;
        else if (t.code == TypeCode::Float) value.f32 = (float)raw_val; // casting uint to float might be wrong if raw_val is bits
    }

    uint16_t as_uint16() const { return value.u16; }
    int32_t as_int32() const { return value.i32; }
    float as_float() const { return value.f32; }
};

} // namespace PE
