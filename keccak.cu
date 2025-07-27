// keccak.cu — Minimal Keccak256 implementation for CUDA
// This uses a simple SHA3 (Keccak) implementation adapted for CUDA
device and kernel usage

#include "keccak.cuh"
#include <stdint.h>

#define ROL64(a, offset) (((a) << (offset)) ^ ((a) >> (64 - (offset))))

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

__device__ void keccakf(uint64_t state[25]) {
    int x, y;
    uint64_t temp, bc[5];

    for (int round = 0; round < 24; ++round) {
        // Theta
        for (x = 0; x < 5; x++)
            bc[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];

        for (x = 0; x < 5; x++) {
            temp = bc[(x + 4) % 5] ^ ROL64(bc[(x + 1) % 5], 1);
            for (y = 0; y < 25; y += 5)
                state[y + x] ^= temp;
        }

        // Rho and Pi
        uint64_t t = state[1];
        int j = 0;
        static const int keccakf_rotc[24] = {
            1, 3, 6, 10, 15, 21, 28, 36, 45, 55,
            2, 14, 27, 41, 56, 8, 25, 43, 62, 18,
            39, 61, 20, 44
        };
        static const int keccakf_piln[24] = {
            10, 7, 11, 17, 18, 3, 5, 16, 8, 21,
            24, 4, 15, 23, 19, 13, 12, 2, 20, 14,
            22, 9, 6, 1
        };
        for (int i = 0; i < 24; i++) {
            j = keccakf_piln[i];
            bc[0] = state[j];
            state[j] = ROL64(t, keccakf_rotc[i]);
            t = bc[0];
        }

        // Chi
        for (y = 0; y < 25; y += 5) {
            for (x = 0; x < 5; x++)
                bc[x] = state[y + x];
            for (x = 0; x < 5; x++)
                state[y + x] ^= (~bc[(x + 1) % 5]) & bc[(x + 2) % 5];
        }

        // Iota
        state[0] ^= keccakf_rndc[round];
    }
}

__device__ void keccak256(const uint8_t* input, size_t inlen, uint8_t* out) {
    uint64_t state[25] = {0};

    // Absorb input
    for (size_t i = 0; i < inlen; i++) {
        ((uint8_t*)state)[i] ^= input[i];
    }

    // Padding
    ((uint8_t*)state)[inlen] ^= 0x01;
    ((uint8_t*)state)[135] ^= 0x80; // 136-byte block for SHA3-256

    keccakf(state);

    // Squeeze output (32 bytes)
    for (int i = 0; i < 32; i++)
        out[i] = ((uint8_t*)state)[i];
}
