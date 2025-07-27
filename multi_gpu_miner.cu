// multi_gpu_miner.cu

extern "C" {
#include <stdint.h>
}

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <thread>
#include <vector>
#include <atomic>

#define HASH_SIZE 32
#define INPUT_SIZE 40
#define BLOCK_SIZE 1024
#define GRID_SIZE 65535

__device__ void keccak256(const uint8_t *input, size_t input_len, uint8_t *output);

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

    uint8_t input[INPUT_SIZE];
    uint8_t hash[HASH_SIZE];

    for (int i = 0; i < 32; ++i)
        input[i] = prev_hash[i];

    for (int i = 0; i < 8; ++i)
        input[32 + i] = (nonce >> ((7 - i) * 8)) & 0xff;

    keccak256(input, INPUT_SIZE, hash);

    if (is_hash_less(hash, max_value)) {
        if (atomicCAS(found_flag, 0, 1) == 0) {
            *found_nonce = nonce;
        }
    }
}

extern "C" void keccak_miner_multi_gpu(
    uint8_t *prev_hash,
    uint8_t *max_value,
    uint64_t start_nonce,
    uint64_t *found_nonce,
    int *found_flag
) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    std::atomic<bool> found(false);
    std::vector<std::thread> threads;

    for (int dev = 0; dev < device_count; ++dev) {
        threads.emplace_back([=, &found]() {
            cudaSetDevice(dev);

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

            uint64_t offset_nonce = start_nonce + (uint64_t)dev * BLOCK_SIZE * GRID_SIZE;

            keccak_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(
                d_prev, d_max, offset_nonce, d_found_nonce, d_found_flag
            );

            cudaDeviceSynchronize();

            int host_flag = 0;
            cudaMemcpy(&host_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
            if (host_flag == 1 && !found.exchange(true)) {
                cudaMemcpy(found_nonce, d_found_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                *found_flag = 1;
            }

            cudaFree(d_prev);
            cudaFree(d_max);
            cudaFree(d_found_nonce);
            cudaFree(d_found_flag);
        });
    }

    for (auto &t : threads) {
        t.join();
    }
}
