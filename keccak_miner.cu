#include <stdint.h>

#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

// Keccak constants
__constant__ uint64_t keccakf_rndc[24] = {
  0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
  0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
  0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
  0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
  0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
  0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
  0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
  0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__constant__ int keccakf_rotc[24] = {
  1,  3,  6, 10, 15, 21,
  28, 36, 45, 55, 2, 14,
  27, 41, 56, 8, 25, 43,
  62, 18, 39, 61, 20, 44
};

__constant__ int keccakf_piln[24] = {
  10, 7, 11, 17, 18, 3,
  5, 16, 8, 21, 24, 4,
  15, 23, 19, 13, 12, 2,
  20, 14, 22, 9, 6, 1
};

// Keccak-f[1600] permutation on state
__device__ void keccakf(uint64_t state[25]) {
  int i, j, round;
  uint64_t t, bc[5];

  for (round = 0; round < 24; round++) {
    // Theta
    for (i = 0; i < 5; i++)
      bc[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
    for (i = 0; i < 5; i++) {
      t = bc[(i + 4) % 5] ^ ROTL64(bc[(i + 1) % 5], 1);
      for (j = 0; j < 25; j += 5)
        state[j + i] ^= t;
    }
    // Rho and pi
    t = state[1];
    for (i = 0; i < 24; i++) {
      j = keccakf_piln[i];
      bc[0] = state[j];
      state[j] = ROTL64(t, keccakf_rotc[i]);
      t = bc[0];
    }
    // Chi
    for (j = 0; j < 25; j += 5) {
      for (i = 0; i < 5; i++)
        bc[i] = state[j + i];
      for (i = 0; i < 5; i++)
        state[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
    }
    // Iota
    state[0] ^= keccakf_rndc[round];
  }
}

// Pad input, absorb, and squeeze keccak256 digest for 64 bytes input
// Input: 64 bytes = 32 bytes nonce + 32 bytes prev_hash
// Output: 32 bytes hash
__device__ void keccak256(const uint8_t *input, uint8_t *output) {
  uint64_t state[25] = {0};
  int i;

  // Absorb phase (rate = 1088 bits = 136 bytes)
  // Input length 64 bytes < 136, so pad input directly
  // Copy input to state lane-wise (little endian)
  for (i = 0; i < 8; i++) {
    state[i] = ((uint64_t*)input)[i];
  }
  // Padding: Append 0x01 at input end and 0x80 at last byte of block (multi-rate padding)
  ((uint8_t*)state)[64] = 0x01;
  ((uint8_t*)state)[135] |= 0x80;

  // Apply Keccak-f permutation
  keccakf(state);

  // Squeeze phase: output first 32 bytes (256 bits) from state
  for (i = 0; i < 4; i++) {
    ((uint64_t*)output)[i] = state[i];
  }
}

// Compare 32-byte hash with max_value (big-endian) and check hash <= max_value
// Return 1 if hash <= max_value else 0
__device__ int check_max_value(const uint8_t *hash, const uint8_t *max_value) {
  for (int i = 0; i < 32; i++) {
    if (hash[i] < max_value[i]) return 1;
    if (hash[i] > max_value[i]) return 0;
  }
  return 1; // equal
}

// Kernel:
// For each thread, try nonce = start_nonce + threadIdx + blockIdx * blockDim
// Hash nonce||prev_hash
// If hash <= max_value, write nonce to output and set *found flag to 1
extern "C"
__global__ void keccak_miner(const uint8_t *prev_hash, const uint8_t *max_value, uint64_t start_nonce, uint64_t *found_nonce, int *found) {
  if (*found) return;  // early exit if found

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t nonce = start_nonce + idx;

  uint8_t input[64];
  // Copy nonce (big endian) into first 32 bytes (zero padded except last 8 bytes)
  for (int i = 0; i < 24; i++) input[i] = 0;
  for (int i = 0; i < 8; i++) input[31 - i] = (nonce >> (8 * i)) & 0xFF;
  // Copy prev_hash (32 bytes) after nonce
  for (int i = 0; i < 32; i++) input[32 + i] = prev_hash[i];

  uint8_t hash[32];
  keccak256(input, hash);

  if (check_max_value(hash, max_value)) {
    *found_nonce = nonce;
    *found = 1;
  }
}
