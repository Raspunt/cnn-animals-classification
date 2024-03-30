import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms

from nn.SimpleCNN import SimpleCNN


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = SimpleCNN().to(device)


def prepare_optimizer() -> tuple:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    return (criterion, optimizer)


def frame_to_tensor(frame):

    img = Image.fromarray(frame)
    img = transform(img)

    img_tensor = img.unsqueeze(0).to(device)
    return img_tensor
