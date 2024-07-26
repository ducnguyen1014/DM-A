import hashlib

def calculate_sha256(data):
    # Convert data to bytes if it’s not already
    if isinstance(data, str):
        data = data.encode()

    # Calculate SHA-256 hash
    sha256_hash = hashlib.sha256(data).hexdigest()

    return sha256_hash

def main():
    # Example usage:
    input_data = "HELLO WORLD"
    hash_value = calculate_sha256(input_data)
    print("INPUT DATA")
    print(input_data)
    print()
    print("OUTPUT HASH SHA-256")
    print(hash_value)


if __name__ == "__main__":
    main()




