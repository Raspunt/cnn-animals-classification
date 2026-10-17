import torch
from torchvision import transforms

# Stored with weights so incompatible preprocessing fails early.
PREPROCESSING = "rgb-32x32-tensor-v1"
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
