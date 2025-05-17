import numpy as np
import os

file_path = "output/resume_embeddings.npy"  # Adjust path as needed

print(f"File exists: {os.path.exists(file_path)}")
print(f"File size: {os.path.getsize(file_path)} bytes")

# Try to determine the format
with open(file_path, 'rb') as f:
    header = f.read(10)
    print(f"File header (hex): {header.hex()}")

try:
    # Try loading with different options
    print("\nTrying to load with allow_pickle=False:")
    data = np.load(file_path, allow_pickle=False)
    print(f"Success! Shape: {data.shape}, dtype: {data.dtype}")
except Exception as e:
    print(f"Error: {str(e)}")

try:
    print("\nTrying to load with allow_pickle=True:")
    data = np.load(file_path, allow_pickle=True)
    print(f"Success! Shape: {data.shape}, dtype: {data.dtype}")
except Exception as e:
    print(f"Error: {str(e)}")