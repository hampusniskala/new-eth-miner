// keccak_miner.cu

#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>
#include "keccak.cuh"

extern "C" {

// CUDA kernel that uses keccak256
__global__ void keccak_kernel(const uint8_t* input, size_t length, uint8_t* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        keccak256(input, length, output);
    }
}

// C-callable launch wrapper
void launch_keccak(const uint8_t* input, size_t length, uint8_t* output) {
    uint8_t *d_input, *d_output;
    cudaMalloc(&d_input, length);
    cudaMemcpy(d_input, input, length, cudaMemcpyHostToDevice);
    cudaMalloc(&d_output, 32);

    keccak_kernel<<<1, 1>>>(d_input, length, d_output);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, 32, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

} // extern "C"

// ========== REAL keccak256 GPU implementation ==========

__device__ __constant__ uint64_t keccakf_rndc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ inline uint64_t ROTL64(uint64_t x, int y) {
    return (x << y) | (x >> (64 - y));
}

__device__ void keccakf(uint64_t st[25]) {
    const int r[24] = {
         1,  3,  6, 10, 15, 21, 28, 36, 45, 55,
         2, 14, 27, 41, 56,  8, 25, 43, 62, 18,
        39, 61, 20, 44
    };
    const int p[24] = {
        10,  7, 11, 17, 18, 3, 5, 16, 8, 21,
        24, 4, 15, 23, 19, 13, 12, 2, 20, 14,
        22,  9, 6,  1
    };

    for (int i = 0; i < 24; ++i) {
        uint64_t bc[5] = {0};

        for (int j = 0; j < 5; ++j)
            bc[j] = st[j] ^ st[j + 5] ^ st[j + 10] ^ st[j + 15] ^ st[j + 20];

        for (int j = 0; j < 5; ++j) {
            uint64_t t = bc[(j + 4) % 5] ^ ROTL64(bc[(j + 1) % 5], 1);
            for (int k = 0; k < 25; k += 5)
                st[j + k] ^= t;
        }

        uint64_t t = st[1];
        for (int j = 0; j < 24; ++j) {
            int pj = p[j];
            uint64_t tmp = st[pj];
            st[pj] = ROTL64(t, r[j]);
            t = tmp;
        }

        for (int j = 0; j < 25; j += 5) {
            uint64_t temp[5];
            for (int k = 0; k < 5; ++k)
                temp[k] = st[j + k];
            for (int k = 0; k < 5; ++k)
                st[j + k] ^= (~temp[(k + 1) % 5]) & temp[(k + 2) % 5];
        }

        st[0] ^= keccakf_rndc[i];
    }
}

__device__ void keccak256(const uint8_t* input, size_t len, uint8_t* output) {
    const size_t rate = 136;
    uint64_t state[25] = {0};

    for (size_t i = 0; i < len; ++i)
        ((uint8_t*)state)[i] ^= input[i];

    ((uint8_t*)state)[len] ^= 0x01;
    ((uint8_t*)state)[rate - 1] ^= 0x80;

    keccakf(state);

    for (int i = 0; i < 32; ++i)
        output[i] = ((uint8_t*)state)[i];
}
