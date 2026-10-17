import random
import uuid
from datetime import datetime
from pathlib import Path

import torch
from matplotlib.figure import Figure
from torch import nn
from torch.utils.data import DataLoader

from db.crud.crud_traning_model import CrudTrainingModel
from db.models import TrainingModel
from nn import device
from nn.dataset_reader import AnimalDataset, DatasetReader
from nn.evaluation import evaluate
from nn.nn_file_manager import NNFileManager
from nn.simple_cnn import SimpleCNN
from settings import settings


class NNTrainer:
    def __init__(self):
        self.nnfm = NNFileManager()

    def train(self, train_dataset, full_dataset, labels, classes):
        random.seed(settings.seed)
        torch.manual_seed(settings.seed)
        class_to_idx = {name: index for index, name in enumerate(classes)}
        label_by_path = dict(zip(full_dataset, labels, strict=True))
        fit_paths, validation_paths = DatasetReader.split(
            train_dataset,
            [label_by_path[path] for path in train_dataset],
            settings.validation_size,
        )
        train_paths = set(train_dataset)
        test_paths = [path for path in full_dataset if path not in train_paths]
        if not test_paths:
            raise ValueError("A separate test split is required before training.")
        root = Path(settings.dataset_folder_path).resolve()
        splits = {
            name: [Path(path).resolve().relative_to(root).as_posix() for path in paths]
            for name, paths in {
                "train": fit_paths,
                "validation": validation_paths,
                "test": test_paths,
            }.items()
        }
        loader_args = dict(
            batch_size=settings.batch_size, num_workers=settings.num_workers
        )
        train_loader = DataLoader(
            AnimalDataset(fit_paths, class_to_idx),
            shuffle=True,
            generator=torch.Generator().manual_seed(settings.seed),
            **loader_args,
        )
        validation_loader = DataLoader(
            AnimalDataset(validation_paths, class_to_idx), **loader_args
        )
        model = SimpleCNN(num_classes=len(classes)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=settings.learning_rate, momentum=settings.momentum
        )
        run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
        run_folder = Path(settings.experiments_folder) / run_id
        training = TrainingModel(
            name="animal CNN",
            start_time=datetime.now(),
            epoch_count=settings.epoch_count,
            learning_rate=settings.learning_rate,
            momentum=settings.momentum,
            model_configuration=str(model),
        )
        history = []
        best_score = (-1.0, float("-inf"))
        print(f"Device: {device}; classes: {classes}")
        print(
            f"Train: {len(fit_paths)}, validation: {len(validation_paths)}, test: {len(test_paths)}"
        )
        for epoch in range(1, settings.epoch_count + 1):
            model.train()
            total_loss = 0.0
            correct = 0
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * targets.size(0)
                correct += (logits.argmax(1) == targets).sum().item()
            validation = evaluate(model, validation_loader, device, len(classes))
            metrics = {
                "train_loss": total_loss / len(fit_paths),
                "train_accuracy": 100 * correct / len(fit_paths),
                "validation_loss": validation.loss,
                "validation_accuracy": validation.accuracy,
            }
            history.append(metrics)
            path = self.nnfm.save_nn(model, classes, epoch, metrics, splits, run_folder)
            score = (validation.accuracy, -validation.loss)
            if score > best_score:
                best_score = score
                self.nnfm.save_best_model(path)
            print(
                f"Epoch {epoch}/{settings.epoch_count}: "
                f"train loss={metrics['train_loss']:.4f}, accuracy={metrics['train_accuracy']:.2f}%; "
                f"validation loss={validation.loss:.4f}, accuracy={validation.accuracy:.2f}%"
            )
        self.nnfm.create_plot_folder()
        plot_path = Path(settings.plot_save_folder) / f"training_{run_id}.png"
        figure = Figure()
        ax = figure.subplots()
        epochs = range(1, settings.epoch_count + 1)
        ax.plot(epochs, [row["train_loss"] for row in history], label="Train")
        ax.plot(epochs, [row["validation_loss"] for row in history], label="Validation")
        ax.set(xlabel="Epoch", ylabel="Cross-entropy loss")
        ax.legend()
        figure.savefig(plot_path)
        training.end_time = datetime.now()
        training.loss_per_epoch = history
        training.loss_plot_path = str(plot_path)
        CrudTrainingModel.create_training(training)
        print(f"Best model by validation: {self.nnfm.best_model_path}")
        return training
