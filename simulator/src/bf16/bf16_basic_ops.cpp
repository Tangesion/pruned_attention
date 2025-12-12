#include "bf16/bf16_basic_ops.h"
#include <cmath>
#include <limits>

namespace bf16 {

void get_bf16_parts(uint16_t bf16, uint16_t& sign, uint16_t& exponent, uint16_t& mantissa) {
    sign = (bf16 >> 15) & 0x1;
    exponent = (bf16 >> 7) & 0xFF;
    mantissa = bf16 & 0x7F;
}

uint16_t compose_bf16(uint16_t sign, uint16_t exponent, uint16_t mantissa) {
    return (sign << 15) | (exponent << 7) | mantissa;
}

float bf16_to_float(uint16_t bf16) {
    uint16_t sign, exponent, mantissa;
    get_bf16_parts(bf16, sign, exponent, mantissa);

    if (exponent == 0) {
        if (mantissa == 0) {
            return sign == 0 ? 0.0f : -0.0f;
        }
        // Subnormal number
        float val = static_cast<float>(mantissa) / 128.0f;
        return (sign == 0 ? val : -val) * std::pow(2.0f, -126);
    } else if (exponent == 0xFF) {
        if (mantissa == 0) {
            return sign == 0 ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
        }
        return std::numeric_limits<float>::quiet_NaN();
    } else {
        // Normalized number
        float val = 1.0f + static_cast<float>(mantissa) / 128.0f;
        return (sign == 0 ? val : -val) * std::pow(2.0f, static_cast<int>(exponent) - 127);
    }
}

uint16_t float_to_bf16(float float_val) {
    if (std::signbit(float_val) && float_val == 0.0f) {
        return 0x8000;
    }
    if (float_val == 0.0f) {
        return 0;
    }
    if (std::isinf(float_val)) {
        return std::signbit(float_val) ? 0xFF80 : 0x7F80;
    }
    if (std::isnan(float_val)) {
        return 0x7FC0;
    }

    uint16_t sign = std::signbit(float_val) ? 1 : 0;
    float_val = std::abs(float_val);

    int exponent = 0;
    float mantissa = float_val;

    if (mantissa >= 2.0f) {
        while (mantissa >= 2.0f) {
            mantissa /= 2.0f;
            exponent++;
        }
    } else if (mantissa < 1.0f && mantissa > 0.0f) {
        while (mantissa < 1.0f) {
            mantissa *= 2.0f;
            exponent--;
        }
    }

    exponent += 127;

    if (exponent < 0) {
        return compose_bf16(sign, 0, 0);
    }
    if (exponent > 255) {
        return compose_bf16(sign, 0xFF, 0);
    }

    mantissa = (mantissa - 1.0f) * 128.0f;
    
    return compose_bf16(sign, exponent, static_cast<uint16_t>(round(mantissa)));
}

uint16_t bf16_add(uint16_t a, uint16_t b) {
    float a_float = bf16_to_float(a);
    float b_float = bf16_to_float(b);
    return float_to_bf16(a_float + b_float);
}

uint16_t bf16_subtract(uint16_t a, uint16_t b) {
    float a_float = bf16_to_float(a);
    float b_float = bf16_to_float(b);
    return float_to_bf16(a_float - b_float);
}

uint16_t bf16_multiply(uint16_t a, uint16_t b) {
    float a_float = bf16_to_float(a);
    float b_float = bf16_to_float(b);
    return float_to_bf16(a_float * b_float);
}

uint16_t bf16_divide(uint16_t a, uint16_t b) {
    float a_float = bf16_to_float(a);
    float b_float = bf16_to_float(b);
    return float_to_bf16(a_float / b_float);
}

uint16_t bf16_sqrt(uint16_t a) {
    float a_float = bf16_to_float(a);
    return float_to_bf16(std::sqrt(a_float));
}

} // namespace bf16
