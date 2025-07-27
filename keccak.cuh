#pragma once
#include <stdint.h>

// Simple placeholder: real Keccak256 implementation needed
__device__ void keccak256(uint8_t* input, size_t input_len, uint8_t* output) {
    // Fill output with dummy hash for testing
    for (int i = 0; i < 32; ++i)
        output[i] = (uint8_t)(input[i % input_len] + i);
}
