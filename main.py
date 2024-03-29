from collections import Counter
import os

import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torch.optim as optim
from torch import nn
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
import cv2

from SimpleCNN import SimpleCNN
from settings import settings


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    # Convert to 3 channel grayscale
    transforms.Grayscale(num_output_channels=3),
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


def test_train_dataset(dataset, labels):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(dataset, labels):
        train_dataset = [dataset[i] for i in train_index]
        test_dataset = [dataset[i] for i in test_index]


    
    return (train_dataset, test_dataset)


def train(train_dataset, full_dataset, labels):
    plt.ion()
    fig, ax = plt.subplots()

    loss_plot = []

    for epoch in range(settings.epoch_count):
        running_loss = 0.0
        for i, img_path in enumerate(train_dataset, 0):
            img = Image.open(img_path)
            img = transform(img)
            img = img.to(device)

            label = torch.tensor(
                labels[full_dataset.index(img_path)], dtype=torch.long)
            label = label.to(device)

            optimizer.zero_grad()
            outputs = net(img.unsqueeze(0))
            loss = criterion(outputs, label.unsqueeze(0))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                loss_plot.append(running_loss / 2000)

                ax.clear()
                ax.plot(loss_plot)
                ax.set_xlabel('Mini-batches')
                ax.set_ylabel('Loss')
                plt.draw()
                plt.pause(0.001)

                running_loss = 0.0

        torch.save(net.state_dict(), settings.trained_model_path)

    plt.ioff()  # Turn off interactive mode


def test(test_dataset, labels, idx_to_class):
    net.load_state_dict(torch.load(settings.trained_model_path))

    class_correct_predictions = {
        class_name: 0 for class_name in idx_to_class.values()}
    class_total_predictions = {
        class_name: 0 for class_name in idx_to_class.values()}

    for i, img_path in enumerate(test_dataset):
        img_raw = Image.open(img_path)
        img = transform(img_raw)
        img = img.to(device)

        true_label = idx_to_class[labels[i]]
        class_total_predictions[true_label] += 1

        if predict(img, idx_to_class) == true_label:
            class_correct_predictions[true_label] += 1

    class_accuracies = {class_name: (class_correct_predictions[class_name] / class_total_predictions[class_name]) if class_total_predictions[class_name] != 0 else 0
                        for class_name in idx_to_class.values()}

    print(f"total predictions:", class_total_predictions)
    print(f"correct predictions:", class_correct_predictions)
    print(
        f"Accuracy on test dataset: {100 * sum(class_correct_predictions.values()) / sum(class_total_predictions.values())} %")

    plt.bar(class_accuracies.keys(), class_accuracies.values())
    plt.xlabel('Animal Class')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy for Each Animal Class')
    plt.show()


def predict(img: Tensor, idx_to_class: list) -> str:
    outputs = net(img)
    _, predicted = torch.max(outputs, 1)

    return idx_to_class[predicted.item()]


def frame_to_tensor(frame):

    img = Image.fromarray(frame)
    img = transform(img)

    img_tensor = img.unsqueeze(0).to(device)
    return img_tensor


def start_webcam(idx_to_class):
    net.load_state_dict(torch.load(settings.trained_model_path))
    vid = cv2.VideoCapture(0)

    while (True):

        ret, frame = vid.read()
        tensor = frame_to_tensor(frame)
        class_prediction = predict(tensor, idx_to_class)
        print(class_prediction)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    (dataset, labels, class_to_idx) = read_datasets()
    (train_dataset, test_dataset) = test_train_dataset(dataset, labels)
    (criterion, optimizer) = prepare_optimizer()
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    class_labels = [idx_to_class[label] for label in labels]
    class_counts = Counter(class_labels)

    print('count of classes')
    print(class_counts)

    # train(train_dataset, dataset, labels)
    test(test_dataset, labels, idx_to_class)

    # start_webcam(idx_to_class)
