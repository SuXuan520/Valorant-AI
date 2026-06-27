import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version (PyTorch built with):", torch.version.cuda if hasattr(torch.version, 'cuda') else "N/A")

import ctypes
import os

# Check nvcuda.dll
try:
    ctypes.WinDLL("nvcuda.dll")
    print("nvcuda.dll: FOUND")
except:
    print("nvcuda.dll: NOT FOUND - driver issue")

# Check CUDA path
cuda_path = os.environ.get("CUDA_PATH", "")
print("CUDA_PATH env:", cuda_path)

# Try to find cudart
try:
    ctypes.CDLL("cudart64_12.dll")
    print("cudart64_12.dll: FOUND")
except:
    print("cudart64_12.dll: NOT FOUND")

try:
    ctypes.CDLL("cudart64_11.dll")
    print("cudart64_11.dll: FOUND")
except:
    print("cudart64_11.dll: NOT FOUND")
