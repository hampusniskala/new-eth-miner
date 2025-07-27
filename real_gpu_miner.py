import os
import sys
import time
import threading
import signal
import numpy as np
import ctypes
from web3 import Web3
import binascii

# Load CUDA shared library
if not os.path.exists('./libkeccak_miner.so'):
    print("[🧪] Shared library libkeccak_miner.so not found. Build it with nvcc.")
    sys.exit(1)

lib = ctypes.CDLL('./libkeccak_miner.so')

# Define argument types for kernel launch
lib.keccak_miner.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # prev_hash
    ctypes.POINTER(ctypes.c_uint8),  # max_value
    ctypes.c_uint64,                 # start_nonce
    ctypes.POINTER(ctypes.c_uint64),# found_nonce
    ctypes.POINTER(ctypes.c_int)    # found flag
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

if BLOCK_SIZE * GRID_SIZE > 2**32:
    raise ValueError("BLOCK_SIZE * GRID_SIZE too large for 32-bit indexing")

def send_test_tx():
    to_address = "0x7DF76FDEedE91d3cB80e4a86158dD9f6D206c98E"
    nonce = w3.eth.get_transaction_count(ADDRESS, "pending")
    base_gas_price = w3.eth.gas_price
    gas_price = int(base_gas_price * 1.1)
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
            time.sleep(2)
        except Exception as e:
            print(f"[!] Error in Mint event listener: {e}")
            time.sleep(5)

def send_mint_tx(value_bytes):
    nonce = w3.eth.get_transaction_count(ADDRESS, "pending")
    base_gas_price = w3.eth.gas_price
    gas_price = int(base_gas_price * 1.1)
    tx = contract.functions.mint(value_bytes).build_transaction({
        "gas": 300000,
        "gasPrice": gas_price,
        "nonce": nonce,
        "chainId": 1,
    })
    signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    print(f"[+] Mint TX sent: https://etherscan.io/tx/{tx_hash.hex()}")

def nonce_to_bytes32(nonce):
    return nonce.to_bytes(32, 'big')

def keccak256_hash(prev_hash_bytes, nonce):
    import sha3
    k = sha3.keccak_256()
    input_bytes = prev_hash_bytes + nonce.to_bytes(8, 'big')
    k.update(input_bytes)
    return k.digest()

def main():
    shared_data = {
        "prev_hash": contract.functions.prev_hash().call(),
        "max_value": contract.functions.max_value().call(),
    }

    prev_hash = shared_data["prev_hash"]
    if isinstance(prev_hash, bytes):
        prev_hash_bytes = prev_hash
    else:
        prev_hash_bytes = bytes.fromhex(prev_hash[2:] if prev_hash.startswith("0x") else prev_hash)

    max_value_int = shared_data["max_value"]
    max_value_bytes = max_value_int.to_bytes(32, 'big')

    found = ctypes.c_int(0)
    found_nonce = ctypes.c_uint64(0)

    prev_hash_c = (ctypes.c_uint8 * 32)(*prev_hash_bytes)
    max_value_c = (ctypes.c_uint8 * 32)(*max_value_bytes)

    start_nonce = 0

    listener_thread = threading.Thread(target=listen_for_mint_event, args=(shared_data,), daemon=True)
    listener_thread.start()

    send_test_tx()

    last_report_time = time.time()
    nonces_checked = 0
    total_nonces_checked = 0
    batch_size = BLOCK_SIZE * GRID_SIZE
    iteration = 0

    try:
        while True:
            found.value = 0
            found_nonce.value = 0

            start_time = time.perf_counter()
            lib.keccak_miner(prev_hash_c, max_value_c, ctypes.c_uint64(start_nonce), ctypes.byref(found_nonce), ctypes.byref(found))
            end_time = time.perf_counter()

            elapsed = end_time - start_time
            if elapsed < 1e-6:
                print("[⚠️] Kernel returned suspiciously fast — skipping this batch.")
                time.sleep(0.1)
                continue

            actual_speed = batch_size / elapsed if elapsed > 0 else 0

            if iteration % 20 == 0:
                print(f"[🧪] Actual GPU kernel speed: {actual_speed:,.0f} hashes/sec (elapsed: {elapsed:.6f} sec)")

            if iteration % 2000 == 0:
                sample_nonce = start_nonce + batch_size // 2
                sample_hash = keccak256_hash(prev_hash_bytes, sample_nonce)
                sample_hash_int = int.from_bytes(sample_hash, 'big')
                print(f"[🔎] Tried Nonce {sample_nonce} resulted in {sample_hash_int} — not valid (max {max_value_int})")

            iteration += 1
            nonces_checked += batch_size
            total_nonces_checked += batch_size

            now = time.time()
            if now - last_report_time >= 5:
                speed = nonces_checked / (now - last_report_time)
                print(f"[⛏️] Speed: {speed:,.0f} nonces/sec | Total tried: {total_nonces_checked:,}")
                last_report_time = now
                nonces_checked = 0

            if found.value:
                print(f"[🌟] Found valid nonce: {found_nonce.value} (Total tried: {total_nonces_checked:,})")
                nonce_bytes = nonce_to_bytes32(found_nonce.value)
                send_mint_tx(nonce_bytes)

                shared_data["prev_hash"] = contract.functions.prev_hash().call()
                shared_data["max_value"] = contract.functions.max_value().call()

                prev_hash = shared_data["prev_hash"]
                prev_hash_bytes = prev_hash if isinstance(prev_hash, bytes) else bytes.fromhex(prev_hash[2:] if prev_hash.startswith("0x") else prev_hash)
                max_value_int = shared_data["max_value"]
                max_value_bytes = max_value_int.to_bytes(32, 'big')

                for i in range(32):
                    prev_hash_c[i] = prev_hash_bytes[i]
                    max_value_c[i] = max_value_bytes[i]

                start_nonce = found_nonce.value + 1
            else:
                start_nonce += batch_size

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user. Stopping mining...")
        stop_flag.set()
        listener_thread.join()
        print(f"[*] Total nonces tried: {total_nonces_checked:,}")

if __name__ == "__main__":
    main()