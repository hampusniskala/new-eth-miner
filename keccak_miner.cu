#include <stdint.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

#define HASH_SIZE 32

// Keccak-f permutation constants
typedef uint64_t uint64;
__device__ __constant__ uint64 keccakf_rndc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ __constant__ int keccakf_rotc[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55,
    2, 14, 27, 41, 56, 8, 25, 43, 62, 18,
    39, 61, 20, 44
};

__device__ __constant__ int keccakf_piln[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21,
    24, 4, 15, 23, 19, 13, 12, 2, 20, 14,
    22, 9, 6, 1
};

__device__ void keccakf(uint64 st[25]) {
    int i, j, round;
    uint64 t, bc[5];
    for (round = 0; round < 24; round++) {
        for (i = 0; i < 5; i++)
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];

        for (i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ __funnelshift_l(bc[(i + 1) % 5], bc[(i + 1) % 5], 1);
            for (j = 0; j < 25; j += 5)
                st[j + i] ^= t;
        }

        t = st[1];
        for (i = 0; i < 24; i++) {
            j = keccakf_piln[i];
            bc[0] = st[j];
            st[j] = __funnelshift_l(t, t, keccakf_rotc[i]);
            t = bc[0];
        }

        for (j = 0; j < 25; j += 5) {
            for (i = 0; i < 5; i++)
                bc[i] = st[j + i];
            for (i = 0; i < 5; i++)
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }

        st[0] ^= keccakf_rndc[round];
    }
}

__device__ void keccak256(const uint8_t *input, size_t inlen, uint8_t *output) {
    uint64 st[25];
    uint8_t temp[200];
    int i;
    for (i = 0; i < 25; i++) st[i] = 0;
    memset(temp, 0, 200);
    memcpy(temp, input, inlen);
    temp[inlen] = 0x01;
    temp[135] |= 0x80;
    for (i = 0; i < 17; i++) {
        for (int j = 0; j < 8; j++) {
            st[i] |= ((uint64)temp[i * 8 + j]) << (8 * j);
        }
    }
    keccakf(st);
    for (i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            output[i * 8 + j] = (uint8_t)(st[i] >> (8 * j));
        }
    }
}

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

    uint8_t input[64];
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
