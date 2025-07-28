#pragma once
#include <stdint.h>
#include <stddef.h>

// Declaration of the real GPU Keccak256 implementation
__device__ void keccak256(const uint8_t* input, size_t input_len, uint8_t* output);
