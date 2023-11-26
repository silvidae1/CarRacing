import torch

# print(f"Used device is {torch.device()}!")
print(f"Is Cuda available {torch.cuda.is_available()}!")

# Use cuda if present
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device available for running: ")
print(device)