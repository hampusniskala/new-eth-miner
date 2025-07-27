import ctypes
import os

# Load the compiled CUDA shared library
lib = ctypes.CDLL('./libkeccak_miner.so')

# Declare argtypes
lib.keccak_miner.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),   # prev_hash
    ctypes.POINTER(ctypes.c_uint8),   # max_value
    ctypes.c_uint64,                  # start_nonce
    ctypes.POINTER(ctypes.c_uint64),  # found_nonce
    ctypes.POINTER(ctypes.c_int)      # found_flag
]

# Fake prev_hash (32 bytes)
prev_hash_bytes = bytes.fromhex("00" * 32)
prev_hash_c = (ctypes.c_uint8 * 32)(*prev_hash_bytes)

# Very high max_value (all F's) to make it easy to find a match
max_value_bytes = bytes.fromhex("ff" * 32)
max_value_c = (ctypes.c_uint8 * 32)(*max_value_bytes)

# Output variables
found_flag = ctypes.c_int(0)
found_nonce = ctypes.c_uint64(0)

# Call kernel
print("[🚀] Launching keccak_miner kernel...")
lib.keccak_miner(
    prev_hash_c,
    max_value_c,
    ctypes.c_uint64(0),        # start_nonce
    ctypes.byref(found_nonce),
    ctypes.byref(found_flag)
)

# Output result
print("[✅] Kernel completed.")
print(f"[🔍] Found flag: {found_flag.value}")
print(f"[🔢] Found nonce: {found_nonce.value}")
