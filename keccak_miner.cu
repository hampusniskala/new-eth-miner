extern "C" {
#include <stdint.h>
}

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

// Keccak-256 constants
#define HASH_SIZE 32

__device__ void keccak256(const uint8_t *input, size_t input_len, uint8_t *output);

// Device comparison: returns true if hash < max
__device__ bool is_hash_less(const uint8_t *hash, const uint8_t *max_value) {
    for (int i = 0; i < HASH_SIZE; ++i) {
        if (hash[i] < max_value[i]) return true;
        if (hash[i] > max_value[i]) return false;
    }
    return false;
}

__global__ void keccak_kernel(
    uint8_t *prev_hash,
    uint8_t *max_value,
    uint64_t start_nonce,
    uint64_t *found_nonce,
    int *found_flag
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = start_nonce + idx;

    if (*found_flag) return;  // early exit

    uint8_t input[40]; // 32 bytes hash + 8 bytes nonce
    uint8_t hash[HASH_SIZE];

    // Copy prev_hash to input
    for (int i = 0; i < 32; ++i)
        input[i] = prev_hash[i];

    // Copy nonce
    for (int i = 0; i < 8; ++i)
        input[32 + i] = (nonce >> ((7 - i) * 8)) & 0xff;

    // Compute hash
    keccak256(input, 40, hash);

    // Compare with max_value
    if (is_hash_less(hash, max_value)) {
        if (atomicCAS(found_flag, 0, 1) == 0) {
            *found_nonce = nonce;
        }
    }
}

// Host wrapper
extern "C" void keccak_miner(
    uint8_t *prev_hash,
    uint8_t *max_value,
    uint64_t start_nonce,
    uint64_t *found_nonce,
    int *found_flag
) {
    // GPU buffers
    uint8_t *d_prev, *d_max;
    uint64_t *d_found_nonce;
    int *d_found_flag;

    cudaMalloc(&d_prev, 32);
    cudaMalloc(&d_max, 32);
    cudaMalloc(&d_found_nonce, sizeof(uint64_t));
    cudaMalloc(&d_found_flag, sizeof(int));

    cudaMemcpy(d_prev, prev_hash, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, max_value, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_found_nonce, found_nonce, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_found_flag, found_flag, sizeof(int), cudaMemcpyHostToDevice);

    // Launch
    dim3 blockSize(256);
    dim3 gridSize(2048);
    keccak_kernel<<<gridSize, blockSize>>>(
        d_prev, d_max, start_nonce, d_found_nonce, d_found_flag
    );
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(found_nonce, d_found_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_prev);
    cudaFree(d_max);
    cudaFree(d_found_nonce);
    cudaFree(d_found_flag);
}
