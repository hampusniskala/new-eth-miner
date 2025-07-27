import os
import sys
import time
import threading
import signal
import numpy as np
import ctypes
from web3 import Web3

# Load CUDA shared library
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

# CUDA kernel launch config
BLOCK_SIZE = 256
GRID_SIZE = 1024

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
    print(f"[ℹ️] Test TX sent: https://etherscan.io/tx/{tx_hash.hex()}")

def listen_for_mint_event(shared_data):
    print("[*] Starting Mint event listener...")
    event_filter = contract.events.Mint.create_filter(from_block='latest')
    while not stop_flag.is_set():
        try:
            for event in event_filter.get_new_entries():
                print(f"[+] Mint event detected: {event}")
                # Update shared data
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

def main():
    shared_data = {
        "prev_hash": contract.functions.prev_hash().call(),
        "max_value": contract.functions.max_value().call(),
    }

    prev_hash = shared_data["prev_hash"]
    # Fix: handle bytes prev_hash correctly
    if isinstance(prev_hash, bytes):
        if prev_hash.startswith(b"0x"):
            prev_hash_bytes = bytes.fromhex(prev_hash[2:].decode())
        else:
            prev_hash_bytes = bytes.fromhex(prev_hash.decode())
    else:
        # If it's already a hex string:
        prev_hash_bytes = bytes.fromhex(prev_hash[2:] if prev_hash.startswith("0x") else prev_hash)

    max_value_int = shared_data["max_value"]
    max_value_bytes = max_value_int.to_bytes(32, 'big')

    found = ctypes.c_int(0)
    found_nonce = ctypes.c_uint64(0)

    # Allocate device memory for prev_hash and max_value
    prev_hash_c = (ctypes.c_uint8 * 32)(*prev_hash_bytes)
    max_value_c = (ctypes.c_uint8 * 32)(*max_value_bytes)

    start_nonce = 0

    # Start Mint event listener thread
    listener_thread = threading.Thread(target=listen_for_mint_event, args=(shared_data,), daemon=True)
    listener_thread.start()

    send_test_tx()

    try:
        while True:
            found.value = 0
            # Call CUDA kernel
            lib.keccak_miner(prev_hash_c, max_value_c, ctypes.c_uint64(start_nonce), ctypes.byref(found_nonce), ctypes.byref(found))
            if found.value:
                print(f"[+] Found valid nonce: {found_nonce.value}")
                nonce_bytes = nonce_to_bytes32(found_nonce.value)
                send_mint_tx(nonce_bytes)
                # After mint, update prev_hash and max_value from contract
                shared_data["prev_hash"] = contract.functions.prev_hash().call()
                shared_data["max_value"] = contract.functions.max_value().call()

                prev_hash = shared_data["prev_hash"]
                if isinstance(prev_hash, bytes):
                    if prev_hash.startswith(b"0x"):
                        prev_hash_bytes = bytes.fromhex(prev_hash[2:].decode())
                    else:
                        prev_hash_bytes = bytes.fromhex(prev_hash.decode())
                else:
                    prev_hash_bytes = bytes.fromhex(prev_hash[2:] if prev_hash.startswith("0x") else prev_hash)

                max_value_int = shared_data["max_value"]
                max_value_bytes = max_value_int.to_bytes(32, 'big')
                for i in range(32):
                    prev_hash_c[i] = prev_hash_bytes[i]
                    max_value_c[i] = max_value_bytes[i]
                start_nonce = found_nonce.value + 1
            else:
                start_nonce += BLOCK_SIZE * GRID_SIZE

    except KeyboardInterrupt:
        print("[*] Interrupted, stopping mining...")
        stop_flag.set()
        listener_thread.join()

if __name__ == "__main__":
    main()
