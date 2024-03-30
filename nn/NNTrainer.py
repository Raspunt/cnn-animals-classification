from datetime import datetime
import uuid
import pickle

import matplotlib.pyplot as plt
import torch
from PIL import Image

from settings import settings
from nn.EpochSaver import EpochSaver
import nn

from db.crud.CrudTrainingModel import CrudTrainingModel
from db.models import TrainingModel


class NNTrainer():

    def __init__(self, optimizer, criterion) -> None:
        self.optimizer = optimizer
        self.criterion = criterion

        self.es = EpochSaver()
        self.crudTraning = CrudTrainingModel()

    def train(self, train_dataset, full_dataset, labels) -> TrainingModel:
        plt.ion()
        fig, ax = plt.subplots()

        loss_plot = []
        traning = TrainingModel()
        traning.name = "traning cnn"
        traning.start_time = datetime.now()
        traning.epoch_count = settings.epoch_count

        for epoch in range(settings.epoch_count):
            running_loss = 0.0
            for i, img_path in enumerate(train_dataset, 0):
                img = Image.open(img_path)
                img = nn.transform(img)
                img = img.to(nn.device)

                label = torch.tensor(
                    labels[full_dataset.index(img_path)], dtype=torch.long)
                label = label.to(nn.device)

                self.optimizer.zero_grad()
                outputs = nn.net(img.unsqueeze(0))
                loss = self.criterion(outputs, label.unsqueeze(0))
                loss.backward()
                self.optimizer.step()
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

            self.es.save_nn(net=nn.net, running_loss=running_loss, epoch=epoch)

        plt.ioff()
        plot_save_path = f"{settings.plot_save_folder}/traning_plot_{str(uuid.uuid4())}.png"
        plt.savefig(plot_save_path)

        traning.loss_per_epoch = pickle.dumps(loss_plot)
        traning.model_configuration = str(nn.net)
        self.crudTraning.create_training(training=traning)

        return traning
