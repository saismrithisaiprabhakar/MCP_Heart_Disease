from cryptography.fernet import Fernet

# Step 1: Generate a new Fernet key (this is your ENCRYPTION KEY)
fernet_key = Fernet.generate_key()
print("🔑 Your Fernet Encryption Key (keep it secret, not in git!):")
print(fernet_key.decode())

# Step 2: Encrypt your GROQ API key
api_key = input("Enter your GROQ_API_KEY: ").strip()
cipher = Fernet(fernet_key)
encrypted_key = cipher.encrypt(api_key.encode())

print("\n🔐 Encrypted API key:")
print(encrypted_key.decode())

# Step 3: Save the encrypted key to a file if you want
with open("encrypted_groq_key.txt", "w") as f:
    f.write(encrypted_key.decode())
print("\n✅ Encrypted key saved to encrypted_groq_key.txt")
