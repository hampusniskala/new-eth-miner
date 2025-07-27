// keccak.cuh — Header for Keccak256 device hashing function

#ifndef KECCAK_CUH
#define KECCAK_CUH

#include <stdint.h>

__device__ void keccak256(const uint8_t* input, size_t input_len, uint8_t* output);

#endif
