#include "bf16/bf16_basic_ops.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <limits>

void test_conversion() {
    std::cout << "Testing float <-> bf16 conversion..." << std::endl;
    float test_val = 3.14159f;
    uint16_t bf16_val = bf16::float_to_bf16(test_val);
    float converted_val = bf16::bf16_to_float(bf16_val);
    assert(std::abs(test_val - converted_val) < 0.01);

    test_val = -2.71828f;
    bf16_val = bf16::float_to_bf16(test_val);
    converted_val = bf16::bf16_to_float(bf16_val);
    assert(std::abs(test_val - converted_val) < 0.01);
    
    std::cout << "Conversion test passed." << std::endl;
}

void test_special_values() {
    std::cout << "Testing special values..." << std::endl;
    assert(bf16::bf16_to_float(0x7F80) == std::numeric_limits<float>::infinity());
    assert(bf16::bf16_to_float(0xFF80) == -std::numeric_limits<float>::infinity());
    assert(std::isnan(bf16::bf16_to_float(0x7FC0)));
    assert(bf16::float_to_bf16(0.0f) == 0x0000);
    assert(bf16::float_to_bf16(-0.0f) == 0x8000);
    std::cout << "Special values test passed." << std::endl;
}

void test_arithmetic() {
    std::cout << "Testing arithmetic operations..." << std::endl;
    uint16_t a_bf16 = bf16::float_to_bf16(2.5f);
    uint16_t b_bf16 = bf16::float_to_bf16(3.5f);

    // Addition
    uint16_t add_res = bf16::bf16_add(a_bf16, b_bf16);
    assert(std::abs(bf16::bf16_to_float(add_res) - 6.0f) < 0.01);

    // Subtraction
    uint16_t sub_res = bf16::bf16_subtract(b_bf16, a_bf16);
    assert(std::abs(bf16::bf16_to_float(sub_res) - 1.0f) < 0.01);

    // Multiplication
    uint16_t mul_res = bf16::bf16_multiply(a_bf16, b_bf16);
    assert(std::abs(bf16::bf16_to_float(mul_res) - 8.75f) < 0.01);

    // Division
    uint16_t div_res = bf16::bf16_divide(b_bf16, a_bf16);
    assert(std::abs(bf16::bf16_to_float(div_res) - 1.4f) < 0.01);

    std::cout << "Arithmetic tests passed." << std::endl;
}

void test_sqrt() {
    std::cout << "Testing sqrt operation..." << std::endl;
    uint16_t val_bf16 = bf16::float_to_bf16(9.0f);
    uint16_t sqrt_res = bf16::bf16_sqrt(val_bf16);
    assert(std::abs(bf16::bf16_to_float(sqrt_res) - 3.0f) < 0.01);
    std::cout << "Sqrt test passed." << std::endl;
}


int main() {
    test_conversion();
    test_special_values();
    test_arithmetic();
    test_sqrt();
    std::cout << "All basic ops tests passed!" << std::endl;
    return 0;
}
