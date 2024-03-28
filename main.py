import os

import torch
import torch.optim as optim
from torch import nn
from PIL import Image
from torchvision import transforms

from SimpleCNN import SimpleCNN
from settings import settings


transform = transforms.Compose([
    transforms.Resize((32, 32)),  
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channel grayscale
    transforms.ToTensor()  # Convert to PyTorch tensor
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = SimpleCNN().to(device)


def prepare_optimizer() -> tuple:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    return (criterion, optimizer)


def read_datasets() -> tuple:
    dataset_raw = os.listdir(settings.dataset_folder_path)
    dataset = []
    labels = []

    class_to_idx = {ds_class: i for i, ds_class in enumerate(dataset_raw)}

    for ds_class in dataset_raw:
        folder_path = f"{settings.dataset_folder_path}/{ds_class}"

        for img in os.listdir(folder_path):
            dataset.append(f"{folder_path}/{img}")
            labels.append(class_to_idx[ds_class])

    return (dataset, labels, class_to_idx)


def test_train_dataset(dataset):
    total_size = len(dataset)
    train_size = int(total_size * 0.8)
    test_size = total_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])
    return (train_dataset, test_dataset)


def train(dataset, labels,train_dataset):


    for epoch in range(settings.epoch_count):
        running_loss = 0.0
        for i, data in enumerate(train_dataset, 0):
            img_path = data
            img = Image.open(img_path)
            img = transform(img)
            img = img.to(device)

            label = torch.tensor(
                labels[dataset.index(img_path)], dtype=torch.long)
            label = label.to(device)

            optimizer.zero_grad()
            outputs = net(img.unsqueeze(0))
            loss = criterion(outputs, label.unsqueeze(0))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
            
        torch.save(net.state_dict(),settings.trained_model_path)


if __name__ == "__main__":
    (dataset, labels, class_to_idx) = read_datasets()
    (train_dataset, test_dataset) = test_train_dataset(dataset)
    (criterion, optimizer) = prepare_optimizer()

    train(dataset,labels,train_dataset)
