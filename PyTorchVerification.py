import torch

# Verify PyTorch installation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("PyTorch version:", torch.__version__)

devNumber = torch.cuda.current_device()
print("Current CUDA device number:", devNumber)

devName = torch.cuda.get_device_name(devNumber)
print("GPU name:", devName)
