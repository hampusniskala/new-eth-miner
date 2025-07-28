#pragma once
#include <stdint.h>
#include <stddef.h>

// Declaration only — the implementation is in keccak_miner.cu
__device__ void keccak256(const uint8_t* input, size_t input_len, uint8_t* output);
