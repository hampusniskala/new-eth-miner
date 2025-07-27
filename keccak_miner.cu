// keccak_miner.cu

#include <stdint.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {

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
    int i, j, round;
    uint64_t t, bc[5];

    for (round = 0; round < 24; round++) {
        for (i = 0; i < 5; i++)
            bc[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
        for (i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ __funnelshift_l(bc[(i + 1) % 5], bc[(i + 1) % 5], 1);
            for (j = 0; j < 25; j += 5)
                state[j + i] ^= t;
        }

        t = state[1];
        i = 0;
        for (int n = 0; n < 24; n++) {
            int r = ((n+1)*(n+2)/2) % 64;
            j = (i * 2 + 3 * i) % 5 + 5 * i;
            bc[0] = state[j];
            state[j] = __funnelshift_l(t, t, r);
            t = bc[0];
            i = j;
        }

        for (j = 0; j < 25; j += 5) {
            for (i = 0; i < 5; i++)
                bc[i] = state[j + i];
            for (i = 0; i < 5; i++)
                state[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }

        state[0] ^= keccakf_rndc[round];
    }
}

__device__ void keccak256(const uint8_t *input, size_t inlen, uint8_t *output) {
    uint64_t state[25];
    for (int i = 0; i < 25; i++)
        state[i] = 0;

    for (size_t i = 0; i < inlen; i++)
        ((uint8_t *)state)[i] ^= input[i];

    state[inlen / 8] ^= 0x01ULL << ((inlen % 8) * 8);
    ((uint8_t *)state)[136 - 1] ^= 0x80;

    keccakf(state);

    for (int i = 0; i < 32; i++)
        output[i] = ((uint8_t *)state)[i];
}

__global__ void keccak_miner(
    const uint8_t *prev_hash,
    const uint8_t *max_value,
    uint64_t start_nonce,
    uint64_t *found_nonce,
    int *found_flag
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = start_nonce + idx;

    if (*found_flag)
        return;

    uint8_t input[32 + 8];
    for (int i = 0; i < 32; i++)
        input[i] = prev_hash[i];

    for (int i = 0; i < 8; i++)
        input[32 + i] = (nonce >> (56 - i * 8)) & 0xff;

    uint8_t hash[32];
    keccak256(input, 40, hash);

    bool is_valid = false;
    for (int i = 0; i < 32; i++) {
        if (hash[i] < max_value[i]) {
            is_valid = true;
            break;
        } else if (hash[i] > max_value[i]) {
            break;
        }
    }

    if (is_valid) {
        if (atomicExch(found_flag, 1) == 0) {
            *found_nonce = nonce;
        }
    }
}

} // extern "C"
