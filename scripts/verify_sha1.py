#!/usr/bin/env python3
"""
SHA-1 Bitcoin Mining Verification Script
Verifies that the GPU is calculating double SHA-1 correctly
"""

import hashlib
import struct
import sys


def sha1(data):
    """Single SHA-1 hash"""
    return hashlib.sha1(data).digest()


def double_sha1(data):
    """Double SHA-1 hash (SHA1(SHA1(data)))"""
    return sha1(sha1(data))


def hex_to_bytes(hex_str):
    """Convert hex string to bytes"""
    return bytes.fromhex(hex_str.replace(' ', ''))


def bytes_to_hex(data):
    """Convert bytes to hex string"""
    return data.hex()


def verify_test_vectors():
    """Test known SHA-1 vectors"""
    print("=== Testing Known Vectors ===\n")

    # Test 1: Zero message
    msg1 = b'\x00' * 32
    hash1 = double_sha1(msg1)
    print(f"Test 1 - Zero message (32 bytes):")
    print(f"Message: {bytes_to_hex(msg1)}")
    print(f"SHA-1^2: {bytes_to_hex(hash1)}")
    print()

    # Test 2: Sequential bytes
    msg2 = bytes(range(32))
    hash2 = double_sha1(msg2)
    print(f"Test 2 - Sequential bytes (0x00-0x1f):")
    print(f"Message: {bytes_to_hex(msg2)}")
    print(f"SHA-1^2: {bytes_to_hex(hash2)}")
    print()

    # Test 3: Bitcoin script puzzle example
    msg3 = bytes(range(32))  # Standard test message
    hash3 = double_sha1(msg3)
    print(f"Test 3 - Standard test message:")
    print(f"Message: {bytes_to_hex(msg3)}")
    print(f"SHA-1^2: {bytes_to_hex(hash3)}")
    print(f"Target for GPU: {bytes_to_hex(hash3)}")
    print()


def verify_collision(msg1_hex, msg2_hex):
    """Verify if two messages produce the same double SHA-1"""
    msg1 = hex_to_bytes(msg1_hex)
    msg2 = hex_to_bytes(msg2_hex)

    hash1 = double_sha1(msg1)
    hash2 = double_sha1(msg2)

    print(f"Message 1: {msg1_hex}")
    print(f"SHA-1^2:   {bytes_to_hex(hash1)}")
    print()
    print(f"Message 2: {msg2_hex}")
    print(f"SHA-1^2:   {bytes_to_hex(hash2)}")
    print()
    print(f"Collision: {'VALID!' if hash1 == hash2 else 'INVALID!'}")

    if hash1 == hash2 and msg1 != msg2:
        print("\n*** SUCCESS: Found a valid SHA-1 collision! ***")
        return True
    return False


def generate_test_cases():
    """Generate test cases for GPU verification"""
    print("=== Test Cases for GPU ===\n")

    # Generate some test messages with known hashes
    test_cases = []

    for i in range(5):
        msg = bytes([j for j in range(32)])
        msg = msg[:-4] + struct.pack('<I', i)  # Modify last 4 bytes
        hash_val = double_sha1(msg)
        test_cases.append((msg, hash_val))

    print("Test messages and their double SHA-1 hashes:")
    for i, (msg, hash_val) in enumerate(test_cases):
        print(f"\nTest {i + 1}:")
        print(f"Message: {bytes_to_hex(msg)}")
        print(f"SHA-1^2: {bytes_to_hex(hash_val)}")

    return test_cases


def check_gpu_output_file(filename="found_collisions.txt"):
    """Parse and verify GPU output file"""
    print(f"\n=== Checking GPU Output File: {filename} ===\n")

    try:
        with open(filename, 'r') as f:
            content = f.read()

        # Parse candidates
        import re
        candidates = re.findall(r'Message: ([0-9a-fA-F]+)', content)
        hashes = re.findall(r'SHA-1\^2: ([0-9a-fA-F]+)', content)

        if candidates:
            print(f"Found {len(candidates)} candidates in file")

            # Verify each one
            for i, (msg_hex, hash_hex) in enumerate(zip(candidates, hashes)):
                msg = hex_to_bytes(msg_hex)
                computed_hash = double_sha1(msg)
                expected_hash = hex_to_bytes(hash_hex)

                print(f"\nCandidate {i + 1}:")
                print(f"Message:  {msg_hex[:32]}...")
                print(f"Expected: {hash_hex}")
                print(f"Computed: {bytes_to_hex(computed_hash)}")
                print(f"Valid:    {'YES' if computed_hash == expected_hash else 'NO'}")

                if computed_hash != expected_hash:
                    print("ERROR: Hash mismatch! GPU might be calculating incorrectly!")
                    return False

            # Check for collisions
            hash_to_msgs = {}
            for msg_hex, hash_hex in zip(candidates, hashes):
                if hash_hex not in hash_to_msgs:
                    hash_to_msgs[hash_hex] = []
                hash_to_msgs[hash_hex].append(msg_hex)

            collisions_found = False
            for hash_hex, msgs in hash_to_msgs.items():
                if len(msgs) > 1:
                    print(f"\n*** COLLISION FOUND for hash {hash_hex} ***")
                    for msg in msgs:
                        print(f"  Message: {msg}")
                    collisions_found = True

            if not collisions_found:
                print("\nNo collisions found yet (all hashes are unique)")

        else:
            print("No candidates found in file")

    except FileNotFoundError:
        print(f"File {filename} not found")
        return False

    return True


def main():
    print("+------------------------------------------+")
    print("|    SHA-1 Mining Python Verification      |")
    print("+------------------------------------------+\n")

    # Run basic scripts
    verify_test_vectors()

    # Generate test cases
    generate_test_cases()

    # Check command line arguments
    if len(sys.argv) == 3:
        print("\n=== Verifying Command Line Collision ===\n")
        verify_collision(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        check_gpu_output_file(sys.argv[1])
    else:
        # Check default output file
        check_gpu_output_file()

        print("\nUsage:")
        print("  python verify_sha1.py                    # Check found_collisions.txt")
        print("  python verify_sha1.py <output_file>      # Check specific file")
        print("  python verify_sha1.py <msg1> <msg2>      # Verify collision pair")


if __name__ == "__main__":
    main()
