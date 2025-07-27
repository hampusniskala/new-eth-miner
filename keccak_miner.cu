#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" __device__ void keccak_256(const uint8_t* input, size_t input_len, uint8_t* hash_output);

extern "C" __global__ void keccak_miner(
    const uint8_t* prev_hash,
    const uint8_t* max_value,
    uint64_t start_nonce,
    uint64_t* found_nonce,
    int* found
) {
    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    int global_id = block_id * blockDim.x + thread_id;
    uint64_t nonce = start_nonce + global_id;

    // Prepare input buffer (32-byte prev_hash + 8-byte nonce)
    uint8_t input[40];
    for (int i = 0; i < 32; ++i) input[i] = prev_hash[i];
    for (int i = 0; i < 8; ++i) input[32 + i] = ((uint8_t*)&nonce)[i];

    uint8_t hash[32];
    keccak_256(input, 40, hash);

    // Check if hash <= max_value
    bool is_valid = true;
    for (int i = 0; i < 32; ++i) {
        if (hash[i] > max_value[i]) {
            is_valid = false;
            break;
        } else if (hash[i] < max_value[i]) {
            break;
        }
    }

    if (is_valid && atomicCAS(found, 0, 1) == 0) {
        *found_nonce = nonce;
    }
}

// --- Minimal Keccak-256 (adapted from tinysha3 or other small impls) ---

#define ROL64(a, offset) (((a) << (offset)) ^ ((a) >> (64 - (offset))))

__device__ __forceinline__ void keccakf(uint64_t st[25]) {
    const uint64_t RC[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
        0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
        0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
        0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
        0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
    };

    const int r[24][5] = {
        { 1,  3,  6, 10, 15}, {28, 36, 45, 55,  2}, {14, 27, 41, 56, 62}, {18, 39, 61, 20, 44},
        { 6, 25,  8, 18, 39}, { 3, 41, 45, 61, 28}, {20, 36, 55,  2, 62}, {14, 27, 44,  1, 10},
        { 6, 15, 25, 56,  3}, {28, 36, 45, 55,  2}, {14, 27, 41, 56, 62}, {18, 39, 61, 20, 44},
        { 6, 25,  8, 18, 39}, { 3, 41, 45, 61, 28}, {20, 36, 55,  2, 62}, {14, 27, 44,  1, 10},
        { 6, 15, 25, 56,  3}, {28, 36, 45, 55,  2}, {14, 27, 41, 56, 62}, {18, 39, 61, 20, 44},
        { 6, 25,  8, 18, 39}, { 3, 41, 45, 61, 28}, {20, 36, 55,  2, 62}, {14, 27, 44,  1, 10}
    };

    for (int round = 0; round < 24; ++round) {
        uint64_t C[5], D[5];
        for (int x = 0; x < 5; ++x) C[x] = st[x] ^ st[x + 5] ^ st[x + 10] ^ st[x + 15] ^ st[x + 20];
        for (int x = 0; x < 5; ++x) D[x] = C[(x + 4) % 5] ^ ROL64(C[(x + 1) % 5], 1);
        for (int x = 0; x < 5; ++x)
            for (int y = 0; y < 5; ++y)
                st[x + 5*y] ^= D[x];

        uint64_t B[25];
        for (int x = 0; x < 5; ++x) {
            for (int y = 0; y < 5; ++y) {
                B[y + ((2*x + 3*y) % 5)*5] = ROL64(st[x + 5*y], r[round][(x + 5*y) % 5]);
            }
        }

        for (int i = 0; i < 25; ++i) st[i] = B[i] ^ ((~B[(i + 1) % 25]) & B[(i + 2) % 25]);
        st[0] ^= RC[round];
    }
}

extern "C" __device__ void keccak_256(const uint8_t* input, size_t input_len, uint8_t* hash_output) {
    uint64_t state[25] = {0};

    for (size_t i = 0; i < input_len; ++i) {
        ((uint8_t*)state)[i] ^= input[i];
    }

    keccakf(state);

    for (int i = 0; i < 32; ++i) {
        hash_output[i] = ((uint8_t*)state)[i];
    }
}