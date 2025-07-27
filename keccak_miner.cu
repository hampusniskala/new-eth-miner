// keccak_miner.cu — Corrected CUDA Keccak256 Miner Kernel

#include <stdint.h>
#include <cuda.h>
#include "keccak.cuh"  // Assume you have a keccak256 device function declared here

extern "C" __global__ void keccak_miner(
    uint8_t* prev_hash,
    uint8_t* max_value,
    uint64_t start_nonce,
    uint64_t* found_nonce,
    int* found
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = start_nonce + idx;

    // Shared memory to short-circuit once a valid nonce is found
    if (*found) return;

    // Construct input: nonce (as 32 bytes) + prev_hash (32 bytes)
    uint8_t input[64];
    for (int i = 0; i < 32; ++i) input[i] = (nonce >> ((31 - i) * 8)) & 0xff;
    for (int i = 0; i < 32; ++i) input[32 + i] = prev_hash[i];

    // Hash the input
    uint8_t hash[32];
    keccak256(input, 64, hash);  // assumes a working keccak256 device function

    // Compare hash against max_value
    bool is_valid = false;
    for (int i = 0; i < 32; ++i) {
        if (hash[i] < max_value[i]) {
            is_valid = true;
            break;
        } else if (hash[i] > max_value[i]) {
            break;
        }
    }

    if (is_valid && atomicCAS(found, 0, 1) == 0) {
        *found_nonce = nonce;
    }
}