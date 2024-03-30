
import torch
from torch import Tensor

from nn import net


def predict(img: Tensor, classes: list) -> str:
    outputs = net(img)
    _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]
