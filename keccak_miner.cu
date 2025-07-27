extern "C" {
#include <stdint.h>
}

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

// Keccak-256 constants
#define HASH_SIZE 32

// Keccak permutation constants
__device__ __constant__ uint64_t keccakf_rndc[24] = {
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
     1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
};

__device__ __constant__ int keccakf_piln[24] = {
    10,  7, 11, 17, 18, 3, 5, 16,
     8, 21, 24, 4, 15, 23, 19, 13,
    12,  2, 20, 14, 22,  9, 6,  1
};

__device__ void keccakf(uint64_t st[25]) {
    int i, j, round;
    uint64_t t, bc[5];

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

__device__ void keccak256(const uint8_t *input, size_t input_len, uint8_t *output) {
    uint64_t st[25];
    uint8_t temp[136];

    for (int i = 0; i < 25; i++)
        st[i] = 0;

    memset(temp, 0, 136);
    memcpy(temp, input, input_len);
    temp[input_len] = 0x01;
    temp[135] |= 0x80;

    for (int i = 0; i < 17; i++) {
        st[i] ^= ((uint64_t*)temp)[i];
    }

    keccakf(st);

    for (int i = 0; i < 4; i++) {
        ((uint64_t*)output)[i] = st[i];
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
    uint8_t *prev_hash,
    uint8_t *max_value,
    uint64_t start_nonce,
    uint64_t *found_nonce,
    int *found_flag
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = start_nonce + idx;

    if (*found_flag) return;

    uint8_t input[40];
    uint8_t hash[HASH_SIZE];

    for (int i = 0; i < 32; ++i)
        input[i] = prev_hash[i];

    for (int i = 0; i < 8; ++i)
        input[32 + i] = (nonce >> ((7 - i) * 8)) & 0xff;

    keccak256(input, 40, hash);

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

    dim3 blockSize(256);
    dim3 gridSize(2048);
    keccak_kernel<<<gridSize, blockSize>>>(
        d_prev, d_max, start_nonce, d_found_nonce, d_found_flag
    );
    cudaDeviceSynchronize();

    cudaMemcpy(found_nonce, d_found_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_prev);
    cudaFree(d_max);
    cudaFree(d_found_nonce);
    cudaFree(d_found_flag);
}
