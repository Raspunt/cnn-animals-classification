from datetime import datetime
import uuid
import pickle

import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from PIL import Image

from settings import settings
from nn.nn_file_manager import NNFileManager
from nn import transform,device,net

from db.crud.crud_traning_model import CrudTrainingModel
from db.models import TrainingModel


class NNTrainer():

    def __init__(self) -> None:
        self.nnfm = NNFileManager()
        self.learning_rate = 0.0001
        self.momentum = 0.9

    def prepare_optimizer(self) -> tuple:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=self.learning_rate, momentum=self.momentum)
        return (criterion, optimizer)

    def train(self, train_dataset, full_dataset, labels):

        (criterion, optimizer) = self.prepare_optimizer()

        plt.ion()
        fig, ax = plt.subplots()

        loss_plot = []
        traning = TrainingModel()
        traning.name = "traning cnn"
        traning.start_time = datetime.now()
        traning.epoch_count = settings.epoch_count
        traning.learning_rate = self.learning_rate
        traning.momentum = self.momentum

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
                    print(
                        f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    loss_plot.append(running_loss / 2000)

                    ax.clear()
                    ax.plot(loss_plot)
                    ax.set_xlabel('Mini-batches')
                    ax.set_ylabel('Loss')
                    plt.draw()
                    plt.pause(0.001)

                    running_loss = 0.0

            self.nnfm.save_nn(net=net, running_loss=running_loss, epoch=epoch)

        plt.ioff()
        plot_save_path = f"{settings.plot_save_folder}/traning_plot_{str(uuid.uuid4())}.png"
 
        plt.savefig(plot_save_path)


        traning.end_time = datetime.now()
        traning.loss_per_epoch = pickle.dumps(loss_plot)
        traning.loss_plot_path = plot_save_path
        traning.model_configuration = str(net)
        CrudTrainingModel.create_training(training=traning)
        plt.close()

        return traning
