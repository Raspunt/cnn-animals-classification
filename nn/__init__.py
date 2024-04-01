import torch
from torchvision import transforms

from nn.simple_cnn import SimpleCNN


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = SimpleCNN().to(device)





