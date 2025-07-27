#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void keccak_kernel(
    uint8_t* prev_hash,
    uint8_t* max_value,
    uint64_t start_nonce,
    uint64_t* found_nonce,
    int* found
) {
    printf("[GPU ✅] Kernel was called! Block %d Thread %d\\n", blockIdx.x, threadIdx.x);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *found = 1;
        *found_nonce = start_nonce + 42;
    }
}

// C-accessible wrapper for Python
extern "C" void keccak_miner(
    uint8_t* prev_hash,
    uint8_t* max_value,
    uint64_t start_nonce,
    uint64_t* found_nonce,
    int* found
) {
    // Launch the kernel
    keccak_kernel<<<1, 1>>>(prev_hash, max_value, start_nonce, found_nonce, found);

    // Wait for GPU to finish
    cudaDeviceSynchronize();
}
