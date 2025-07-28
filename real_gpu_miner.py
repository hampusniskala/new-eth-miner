import os
import sys
import time
import threading
import ctypes
import binascii
import random
from web3 import Web3
import sha3

lib = ctypes.CDLL('./libkeccak_miner.so')

lib.keccak_miner.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_int)
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
GRID_SIZE = 8192
BATCH_SIZE = BLOCK_SIZE * GRID_SIZE

def generate_candidate_values():
    batch = bytearray(BATCH_SIZE * 32)
    for i in range(BATCH_SIZE):
        prefix = os.urandom(13)
        zeros = b'\x00' * 19
        full = prefix + zeros
        batch[i*32:(i+1)*32] = full
    return batch

def keccak256_hash(value, prev_hash):
    k = sha3.keccak_256()
    k.update(value + prev_hash)
    return k.digest()

def send_mint_tx(value_bytes):
    try:
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
    except Exception as e:
        print(f"[!] Error sending mint TX: {e}")

def listen_for_mints(shared_data):
    print("[*] Starting Mint event listener...")
    event_filter = contract.events.Mint.create_filter(from_block='latest')
    while not stop_flag.is_set():
        try:
            for event in event_filter.get_new_entries():
                print(f"[+] Mint detected: {event}")
                shared_data["prev_hash"] = contract.functions.prev_hash().call()
                shared_data["max_value"] = contract.functions.max_value().call()
                print("[*] Updated prev_hash and max_value.")
        except Exception as e:
            print(f"[!] Error in Mint listener: {e}")
        time.sleep(3)

def main():
    try:
        shared_data = {
            "prev_hash": contract.functions.prev_hash().call(),
            "max_value": contract.functions.max_value().call(),
        }
    except Exception as e:
        print(f"[!] Web3 init failed: {e}")
        sys.exit(1)

    listener_thread = threading.Thread(target=listen_for_mints, args=(shared_data,), daemon=True)
    listener_thread.start()

    found = ctypes.c_int(0)
    found_index = ctypes.c_uint64(0)
    total_tries = 0
    iteration = 0

    while True:
        prev_hash = shared_data["prev_hash"]
        max_value = shared_data["max_value"]

        prev_hash_bytes = prev_hash if isinstance(prev_hash, bytes) else bytes.fromhex(prev_hash[2:] if prev_hash.startswith("0x") else prev_hash)
        max_value_bytes = max_value.to_bytes(32, 'big')

        prev_hash_c = (ctypes.c_uint8 * 32)(*prev_hash_bytes)
        max_value_c = (ctypes.c_uint8 * 32)(*max_value_bytes)

        values_batch = generate_candidate_values()
        values_c = (ctypes.c_uint8 * (32 * BATCH_SIZE)).from_buffer_copy(values_batch)

        found.value = 0
        found_index.value = 0


        lib.keccak_miner(values_c, prev_hash_c, max_value_c, ctypes.byref(found_index), ctypes.byref(found))

        print(f"[⛏️] Total tried: {total_tries:,}")

        if iteration % 5 == 0:
            sample = values_batch[0:32]
            sample_hash = keccak256_hash(sample, prev_hash_bytes)
            sample_hash_int = int.from_bytes(sample_hash, 'big')
            is_valid = sample_hash_int <= max_value
            percentage_of_max = (sample_hash_int / max_value) * 100
            print(f"[📡] Sample value={sample.hex()} => hash={sample_hash_int} ({percentage_of_max:.2f}% of max) (valid={is_valid}) max={max_value}")

        if iteration % 10 == 0:
            print(f"[📡] Prev_hash={prev_hash_bytes.hex()}")

        if found.value:
            offset = found_index.value * 32
            winning_value = values_batch[offset:offset+32]
            print(f"[🌟] Found winning value: {winning_value.hex()}")
            send_mint_tx(winning_value)

        total_tries += BATCH_SIZE
        iteration += 1

if __name__ == "__main__":
    main()
