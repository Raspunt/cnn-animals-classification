from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


@torch.inference_mode()
def predict(img: Tensor, classes: list[str], model: nn.Module) -> Prediction:
    if img.ndim == 3:
        img = img.unsqueeze(0)
    if img.ndim != 4 or img.shape[0] != 1:
        raise ValueError("Prediction expects one image: [C, H, W] or [1, C, H, W].")
    model.eval()
    img = img.to(next(model.parameters()).device)
    logits = model(img)
    if logits.shape != (1, len(classes)):
        raise ValueError("The model outputs do not match its saved class mapping.")
    probabilities = logits.softmax(dim=1)[0].cpu().tolist()
    predicted = max(range(len(classes)), key=probabilities.__getitem__)
    return Prediction(
        label=classes[predicted],
        confidence=probabilities[predicted],
        probabilities=dict(zip(classes, probabilities)),
    )
