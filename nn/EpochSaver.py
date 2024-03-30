
import os
import shutil
import glob
from datetime import datetime

import torch

from . import SimpleCNN
from db.models import TrainingModel


class EpochSaver():

    def remove_all_epoch(self, remove_without_dialog: bool = False):

        if remove_without_dialog:
            epochs = glob.glob("./out/epoch/*")
            for epoch in epochs:
                shutil.rmtree(epoch)

        else:
            valid_answers = {"yes": True, "y": True,
                             "ye": True, "no": False, "n": False}
            while True:
                print("did you want remove all epoch [y/n]", end=" ")
                choice = input().lower()
                if choice in valid_answers:
                    if valid_answers[choice]:
                        epochs = glob.glob("./out/epoch/*")
                        for epoch in epochs:
                            shutil.rmtree(epoch)
                    return valid_answers[choice]
                else:
                    print(
                        "[!] Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")

    def save_nn(self, epoch: int, running_loss: float, net: SimpleCNN) -> TrainingModel:

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")

        folder_path = f'./out/experiments/experiment_{epoch +1}_{running_loss / 2000}_{current_time}'
        os.makedirs(folder_path, exist_ok=True)
        torch.save(net.state_dict(), f'{folder_path}/trained.pth')

        return
