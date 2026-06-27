import torch
print("PyTorch available:", torch.__version__ if hasattr(torch, '__version__') else "?")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU found, will train on CPU")
