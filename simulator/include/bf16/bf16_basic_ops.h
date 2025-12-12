#ifndef BF16_BASIC_OPS_H
#define BF16_BASIC_OPS_H

#include <cstdint>

namespace bf16 {

// Decomposes a BF16 into its sign, exponent, and mantissa.
void get_bf16_parts(uint16_t bf16, uint16_t& sign, uint16_t& exponent, uint16_t& mantissa);

// Composes a BF16 from its sign, exponent, and mantissa.
uint16_t compose_bf16(uint16_t sign, uint16_t exponent, uint16_t mantissa);

// Converts a BF16 to a float32.
float bf16_to_float(uint16_t bf16);

// Converts a float32 to a BF16.
uint16_t float_to_bf16(float float_val);

// BF16 addition operation.
uint16_t bf16_add(uint16_t a, uint16_t b);

// BF16 subtraction operation.
uint16_t bf16_subtract(uint16_t a, uint16_t b);

// BF16 multiplication operation.
uint16_t bf16_multiply(uint16_t a, uint16_t b);

// BF16 division operation.
uint16_t bf16_divide(uint16_t a, uint16_t b);

// BF16 square root operation.
uint16_t bf16_sqrt(uint16_t a);

} // namespace bf16

#endif // BF16_BASIC_OPS_H
