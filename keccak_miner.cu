#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "keccak.cuh"

__global__ void keccak_kernel(
    uint8_t* prev_hash,
    uint8_t* max_value,
    uint64_t start_nonce,
    uint64_t* found_nonce,
    int* found
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = start_nonce + idx;

    // Input: nonce (8 bytes) + prev_hash (32 bytes)
    uint8_t input[40];
    for (int i = 0; i < 8; ++i)
        input[i] = (nonce >> (8 * (7 - i))) & 0xff;
    for (int i = 0; i < 32; ++i)
        input[8 + i] = prev_hash[i];

    uint8_t hash[32];
    keccak256(input, 40, hash);

    // Compare against max_value
    bool valid = true;
    for (int i = 0; i < 32; ++i) {
        if (hash[i] > max_value[i]) {
            valid = false;
            break;
        } else if (hash[i] < max_value[i]) {
            break;
        }
    }

    if (valid && atomicCAS(found, 0, 1) == 0) {
        *found_nonce = nonce;
        printf("[GPU ✅] Found valid nonce: %llu\n", nonce);
    }
}

extern "C" void keccak_miner(
    uint8_t* prev_hash,
    uint8_t* max_value,
    uint64_t start_nonce,
    uint64_t* found_nonce,
    int* found
) {
    uint8_t *d_prev_hash, *d_max_value;
    uint64_t *d_found_nonce;
    int *d_found;

    cudaMalloc(&d_prev_hash, 32);
    cudaMalloc(&d_max_value, 32);
    cudaMalloc(&d_found_nonce, sizeof(uint64_t));
    cudaMalloc(&d_found, sizeof(int));

    cudaMemcpy(d_prev_hash, prev_hash, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_value, max_value, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_found_nonce, found_nonce, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_found, found, sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = 1024;

    keccak_kernel<<<blocks, threads>>>(d_prev_hash, d_max_value, start_nonce, d_found_nonce, d_found);
    cudaDeviceSynchronize();

    cudaMemcpy(found_nonce, d_found_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(found, d_found, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_prev_hash);
    cudaFree(d_max_value);
    cudaFree(d_found_nonce);
    cudaFree(d_found);
}
