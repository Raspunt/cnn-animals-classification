from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Evaluation:
    loss: float
    accuracy: float
    confusion_matrix: torch.Tensor


@torch.inference_mode()
def evaluate(model, loader, device, num_classes) -> Evaluation:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        predictions = logits.argmax(dim=1)
        indices = (labels * num_classes + predictions).cpu()
        confusion += torch.bincount(indices, minlength=num_classes**2).reshape(
            num_classes, num_classes
        )
    total = confusion.sum().item()
    if total == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    return Evaluation(
        loss=total_loss / total,
        accuracy=100 * confusion.diag().sum().item() / total,
        confusion_matrix=confusion,
    )
