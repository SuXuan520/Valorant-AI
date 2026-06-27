import subprocess, sys

# nvidia-smi
try:
    out = subprocess.check_output(["nvidia-smi"], text=True)
    print(out[:1500])
except FileNotFoundError:
    print("nvidia-smi not found — no NVIDIA driver or no NVIDIA GPU")
except Exception as e:
    print("Error:", e)
