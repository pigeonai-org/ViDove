import argparse
from cryptography.fernet import Fernet
"""
    encrypt_apikey.py: Encrypt apikey for vidove public demo
    Usage:
        python encrypt_apikey.py --g # Generate a new VIDOVE_DECODE_KEY
        python encrypt_apikey.py -e --key [VIDOVE_DECODE_KEY] --data [API_KEY] # encrypt API_KEY
        python encrypt_apikey.py -d --key [VIDOVE_DECODE_KEY] --en_data [encode_data] # decrypt encode_data
"""

# Function to generate a new key
def generate_key():
    key = Fernet.generate_key()
    print(key.decode())

# Function to encrypt a message
def encrypt_message(key, message):
    fernet = Fernet(key)
    encMessage = fernet.encrypt(message.encode())
    print("Encrypted message:", encMessage.decode())

# Function to decrypt a message
def decrypt_message(key, encMessage):
    fernet = Fernet(key)
    decMessage = fernet.decrypt(encMessage.encode()).decode()
    print("Decrypted message:", decMessage)

def main():
    parser = argparse.ArgumentParser(description="Encrypt or decrypt messages using Fernet")
    parser.add_argument('-g', '--generate', action='store_true', help='Generate a new encryption key')
    parser.add_argument('-e', '--encrypt', action='store_true', help='Encrypt a message')
    parser.add_argument('-d', '--decrypt', action='store_true', help='Decrypt a message')
    parser.add_argument('--key', type=str, help='The encryption key')
    parser.add_argument('--data', type=str, help='The data to encrypt')
    parser.add_argument('--en_data', type=str, help='The data to decrypt')

    args = parser.parse_args()

    if args.generate:
        generate_key()
    elif args.encrypt and args.key and args.data:
        encrypt_message(args.key, args.data)
    elif args.decrypt and args.key and args.en_data:
        decrypt_message(args.key, args.en_data)
    else:
        print("Invalid or insufficient arguments provided.")
        parser.print_help()

if __name__ == "__main__":
    main()
