"""
Test GPU usage during YOLO inference
"""
import torch, time, os
from ultralytics import YOLO

# 1) Check CUDA
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

# 2) Quick inference test on GPU
model = YOLO("D:/aidemo/Valorant-Enemy-Detection-with-YOLO11-main/exported_models/original_best.pt")
print("Model loaded")

# Create a dummy 640x640 image
import numpy as np
dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# Warm up
for _ in range(5):
    model(dummy, device=0, verbose=False)

# Timed run
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    model(dummy, device=0, verbose=False)
torch.cuda.synchronize()
elapsed = time.time() - start

print(f"100 inferences on GPU: {elapsed:.2f}s ({100/elapsed:.1f} FPS)")

# 3) Check if model is on GPU
print("Model device:", next(model.model.parameters()).device)

# 4) Show GPU memory
print(f"GPU memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")
print(f"GPU memory cached:    {torch.cuda.memory_reserved(0)/1024**2:.1f} MB")
