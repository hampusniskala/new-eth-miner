import ctypes

# Load the shared CUDA library
lib = ctypes.CDLL('./libkeccak_miner.so')

# Define argument types
lib.keccak_miner.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # prev_hash
    ctypes.POINTER(ctypes.c_uint8),  # max_value
    ctypes.c_uint64,                 # start_nonce
    ctypes.POINTER(ctypes.c_uint64),# found_nonce
    ctypes.POINTER(ctypes.c_int)    # found flag
]

# Allocate dummy inputs
prev_hash = (ctypes.c_uint8 * 32)(*([0] * 32))
max_value = (ctypes.c_uint8 * 32)(*([255] * 32))
found_nonce = ctypes.c_uint64(0)
found = ctypes.c_int(0)

# Call the kernel with 1 thread
lib.keccak_miner(prev_hash, max_value, ctypes.c_uint64(123456), ctypes.byref(found_nonce), ctypes.byref(found))

print("[Python] Kernel call completed.")
print(f"[Python] Found: {found.value} | Nonce: {found_nonce.value}")
