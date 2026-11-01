import torch
import torchvision

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
  print("GPU:", torch.cuda.get_device_name(0))
  print("CUDA version:", torch.version.cuda)
  print("GPU memory:", round(
    torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
  ), "GB")

  x = torch.randn(1000, 1000, device="cuda")
  y = x @ x
  print("GPU computation successful:", y.device)
else:
  print("CUDA is unavailable. Check the PyTorch installation.")