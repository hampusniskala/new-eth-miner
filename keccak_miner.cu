extern "C" {
#include <stdint.h>
}

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define HASH_SIZE 32

// Device-side Keccak implementation (include full implementation or link against one)
__device__ void keccak256(const uint8_t *input, size_t input_len, uint8_t *output);

// Comparison helper: returns true if hash < max_value
__device__ bool is_hash_less(const uint8_t *hash, const uint8_t *max_value) {
    for (int i = 0; i < HASH_SIZE; ++i) {
        if (hash[i] < max_value[i]) return true;
        if (hash[i] > max_value[i]) return false;
    }
    return false;
}

__global__ void keccak_kernel(
    const uint8_t *prev_hash,
    const uint8_t *max_value,
    uint64_t start_nonce,
    uint64_t *found_nonce,
    int *found_flag
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = start_nonce + idx;

    if (*found_flag) return;

    uint8_t input[64]; // 32-byte nonce + 32-byte prev_hash
    uint8_t hash[HASH_SIZE];

    // Convert nonce to 32-byte big-endian
    for (int i = 0; i < 32; ++i)
        input[i] = (i < 24) ? 0 : (nonce >> ((31 - i) * 8)) & 0xff;

    // Append prev_hash
    for (int i = 0; i < 32; ++i)
        input[32 + i] = prev_hash[i];

    keccak256(input, 64, hash);

    if (is_hash_less(hash, max_value)) {
        if (atomicCAS(found_flag, 0, 1) == 0) {
            *found_nonce = nonce;
        }
    }
}

extern "C" void keccak_miner(
    uint8_t *prev_hash,
    uint8_t *max_value,
    uint64_t start_nonce,
    uint64_t *found_nonce,
    int *found_flag
) {
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

    dim3 blockSize(512);
    dim3 gridSize(4096);

    keccak_kernel<<<gridSize, blockSize>>>(d_prev, d_max, start_nonce, d_found_nonce, d_found_flag);
    cudaDeviceSynchronize();

    cudaMemcpy(found_nonce, d_found_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_prev);
    cudaFree(d_max);
    cudaFree(d_found_nonce);
    cudaFree(d_found_flag);
}
