#include <stdint.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

#define HASH_SIZE 32

__device__ void keccak256(const uint8_t *input, size_t input_len, uint8_t *output);

__device__ bool is_hash_less(const uint8_t *hash, const uint8_t *max_value) {
    for (int i = 0; i < HASH_SIZE; ++i) {
        if (hash[i] < max_value[i]) return true;
        if (hash[i] > max_value[i]) return false;
    }
    return false;
}

__global__ void keccak_kernel(
    uint8_t *values,
    uint8_t *prev_hash,
    uint8_t *max_value,
    uint64_t *found_index,
    int *found_flag
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (*found_flag) return;

    uint8_t input[64]; // 32 bytes value + 32 bytes prev_hash
    uint8_t hash[HASH_SIZE];

    for (int i = 0; i < 32; ++i)
        input[i] = values[idx * 32 + i];
    for (int i = 0; i < 32; ++i)
        input[32 + i] = prev_hash[i];

    keccak256(input, 64, hash);

    if (is_hash_less(hash, max_value)) {
        if (atomicCAS(found_flag, 0, 1) == 0) {
            *found_index = idx;
        }
    }
}

extern "C" void keccak_miner(
    uint8_t *values,
    uint8_t *prev_hash,
    uint8_t *max_value,
    uint64_t *found_index,
    int *found_flag
) {
    uint8_t *d_values, *d_prev_hash, *d_max_value;
    uint64_t *d_found_index;
    int *d_found_flag;

    size_t batch_size = 512 * 4096;
    cudaMalloc(&d_values, 32 * batch_size);
    cudaMalloc(&d_prev_hash, 32);
    cudaMalloc(&d_max_value, 32);
    cudaMalloc(&d_found_index, sizeof(uint64_t));
    cudaMalloc(&d_found_flag, sizeof(int));

    cudaMemcpy(d_values, values, 32 * batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_hash, prev_hash, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_value, max_value, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_found_index, found_index, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_found_flag, found_flag, sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(512);
    dim3 gridSize(4096);
    keccak_kernel<<<gridSize, blockSize>>>(d_values, d_prev_hash, d_max_value, d_found_index, d_found_flag);
    cudaDeviceSynchronize();

    cudaMemcpy(found_index, d_found_index, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_values);
    cudaFree(d_prev_hash);
    cudaFree(d_max_value);
    cudaFree(d_found_index);
    cudaFree(d_found_flag);
}
