import os
import sys
import time
import threading
import signal
import numpy as np
import ctypes
import binascii
from web3 import Web3
from secrets import token_bytes

# Load CUDA shared library
if not os.path.exists('./libkeccak_miner.so'):
    print("[🧪] Shared library libkeccak_miner.so not found. Build it with nvcc.")
    sys.exit(1)

lib = ctypes.CDLL('./libkeccak_miner.so')

# Define argument types for kernel launch
lib.keccak_miner.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # values
    ctypes.POINTER(ctypes.c_uint8),  # prev_hash
    ctypes.POINTER(ctypes.c_uint8),  # max_value
    ctypes.POINTER(ctypes.c_uint64), # found_index
    ctypes.POINTER(ctypes.c_int)     # found flag
]

PRIVATE_KEY = os.getenv("PRIVATE_KEY")
ADDRESS = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))
INFURA_URL = os.getenv("INFURA_URL")
CONTRACT_ADDRESS = Web3.to_checksum_address(os.getenv("CONTRACT_ADDRESS"))

ABI = [
    {"constant": True, "name": "prev_hash", "outputs": [{"name": "", "type": "bytes32"}], "type": "function"},
    {"constant": True, "name": "max_value", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
    {"constant": False, "name": "mint", "inputs": [{"name": "value", "type": "bytes32"}], "type": "function"},
    {"anonymous": False, "inputs": [{"indexed": True, "name": "minter", "type": "address"}], "name": "Mint", "type": "event"}
]

w3 = Web3(Web3.HTTPProvider(INFURA_URL))
contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=ABI)
stop_flag = threading.Event()

BLOCK_SIZE = 512
GRID_SIZE = 4096
BATCH_SIZE = BLOCK_SIZE * GRID_SIZE

if BATCH_SIZE > 2**32:
    raise ValueError("BLOCK_SIZE * GRID_SIZE too large for 32-bit indexing")

def send_test_tx():
    to_address = "0x7DF76FDEedE91d3cB80e4a86158dD9f6D206c98E"
    nonce = w3.eth.get_transaction_count(ADDRESS, "pending")
    gas_price = int(w3.eth.gas_price * 1.1)
    tx = {
        "to": to_address,
        "value": 0,
        "gas": 21000,
        "gasPrice": gas_price,
        "nonce": nonce,
        "chainId": 1,
    }
    signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    print(f"[🧪] Test TX sent: https://etherscan.io/tx/{tx_hash.hex()}")

def listen_for_mint_event(shared_data):
    print("[*] Starting Mint event listener...")
    event_filter = contract.events.Mint.create_filter(from_block='latest')
    while not stop_flag.is_set():
        try:
            for event in event_filter.get_new_entries():
                print(f"[+] Mint event detected: {event}")
                shared_data["prev_hash"] = contract.functions.prev_hash().call()
                shared_data["max_value"] = contract.functions.max_value().call()
                print("[*] Updated prev_hash and max_value after Mint event.")
        except Exception as e:
            print(f"[!] Error in Mint event listener: {e}")
        time.sleep(2)

def send_mint_tx(value_bytes):
    nonce = w3.eth.get_transaction_count(ADDRESS, "pending")
    gas_price = int(w3.eth.gas_price * 1.1)
    tx = contract.functions.mint(value_bytes).build_transaction({
        "gas": 300000,
        "gasPrice": gas_price,
        "nonce": nonce,
        "chainId": 1,
    })
    signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    print(f"[+] Mint TX sent: https://etherscan.io/tx/{tx_hash.hex()}")

def main():
    shared_data = {
        "prev_hash": contract.functions.prev_hash().call(),
        "max_value": contract.functions.max_value().call(),
    }

    found = ctypes.c_int(0)
    found_index = ctypes.c_uint64(0)

    listener_thread = threading.Thread(target=listen_for_mint_event, args=(shared_data,), daemon=True)
    listener_thread.start()

    send_test_tx()

    values_buffer = (ctypes.c_uint8 * (32 * BATCH_SIZE))()

    last_report_time = time.time()
    total_checked = 0
    iteration = 0

    try:
        while True:
            prev_hash = shared_data["prev_hash"]
            max_value = shared_data["max_value"]

            if isinstance(prev_hash, bytes):
                prev_hash_bytes = prev_hash
            else:
                prev_hash_bytes = bytes.fromhex(prev_hash[2:] if prev_hash.startswith("0x") else prev_hash)

            max_value_bytes = max_value.to_bytes(32, 'big')

            # Fill random values
            for i in range(BATCH_SIZE):
                value = token_bytes(32)
                for j in range(32):
                    values_buffer[i * 32 + j] = value[j]

            prev_hash_c = (ctypes.c_uint8 * 32)(*prev_hash_bytes)
            max_value_c = (ctypes.c_uint8 * 32)(*max_value_bytes)

            found.value = 0
            found_index.value = 0

            start_time = time.perf_counter()
            lib.keccak_miner(values_buffer, prev_hash_c, max_value_c, ctypes.byref(found_index), ctypes.byref(found))
            end_time = time.perf_counter()

            elapsed = end_time - start_time
            if elapsed < 1e-6:
                print("[⚠️] Kernel returned suspiciously fast — skipping this batch.")
                time.sleep(0.1)
                continue

            if found.value:
                idx = found_index.value
                result = bytes(values_buffer[idx*32:(idx+1)*32])
                print(f"[🌟] Found valid value: {binascii.hexlify(result).decode()} (Index: {idx}, Checked: {total_checked:,})")
                send_mint_tx(result)
            else:
                if iteration % 1000 == 0:
                    print(f"[📡] Current prev_hash: {binascii.hexlify(prev_hash_bytes).decode()}")
                if iteration % 2000 == 0:
                    sample = bytes(values_buffer[32*100:32*101])
                    from Crypto.Hash import keccak
                    k = keccak.new(digest_bits=256)
                    k.update(sample + prev_hash_bytes)
                    result_int = int.from_bytes(k.digest(), 'big')
                    print(f"[🔎] Tried sample hash = {result_int} (max {max_value})")
                if iteration % 8000 == 0:
                    speed = BATCH_SIZE / elapsed
                    print(f"[🧪] Speed: {speed:,.0f} hashes/sec (elapsed: {elapsed:.6f} sec)")

            total_checked += BATCH_SIZE
            iteration += 1

            if time.time() - last_report_time >= 5:
                print(f"[⛏️] Total hashes attempted: {total_checked:,}")
                last_report_time = time.time()

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user. Stopping mining...")
        stop_flag.set()
        listener_thread.join()
        print(f"[*] Total nonces tried: {total_checked:,}")

if __name__ == "__main__":
    main()
