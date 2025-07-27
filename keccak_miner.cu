// keccak_miner.cu — Working GPU Keccak256 nonce miner with debug output

#include <stdint.h>
#include <stdio.h>
#include "keccak.cuh"

extern "C" __global__ void keccak_miner(
    uint8_t* prev_hash,
    uint8_t* max_value,
    uint64_t start_nonce,
    uint64_t* found_nonce,
    int* found
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = start_nonce + idx;

    if (*found) return;  // Skip if already found by another thread

    // Prepare input = nonce (32 bytes) + prev_hash (32 bytes)
    uint8_t input[64];
    for (int i = 0; i < 32; ++i) input[i] = (nonce >> ((31 - i) * 8)) & 0xFF;
    for (int i = 0; i < 32; ++i) input[32 + i] = prev_hash[i];

    // Hash it
    uint8_t hash[32];
    keccak256(input, 64, hash);

    // Print one result for debugging
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[GPU] Nonce: %llu → Hash: %02x%02x%02x%02x...\n", nonce, hash[0], hash[1], hash[2], hash[3]);
    }

    // Compare with max_value
    bool is_valid = true;
    for (int i = 0; i < 32; ++i) {
        if (hash[i] < max_value[i]) {
            break;
        } else if (hash[i] > max_value[i]) {
            is_valid = false;
            break;
        }
    }

    // Save winning nonce atomically
    if (is_valid && atomicCAS(found, 0, 1) == 0) {
        *found_nonce = nonce;
        printf("[🎯 GPU] Valid nonce found: %llu\n", nonce);
    }
}
