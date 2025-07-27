// keccak_miner.cu — Simple test kernel that just prints
#include <stdint.h>
#include <stdio.h>

extern "C" __global__ void keccak_miner(
    uint8_t* prev_hash,
    uint8_t* max_value,
    uint64_t start_nonce,
    uint64_t* found_nonce,
    int* found
) {
    printf("[✅ CUDA] Kernel was actually called! Block %d Thread %d\\n", blockIdx.x, threadIdx.x);

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *found = 1;
        *found_nonce = start_nonce + 123;
    }
}
