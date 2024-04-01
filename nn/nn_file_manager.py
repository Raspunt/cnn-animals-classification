
import os
import shutil
import glob
from datetime import datetime
import shutil
import torch

from . import SimpleCNN
from db.models import TestModel, TrainingModel

from settings import settings


class NNFileManager():
   

    def create_plot_folder(self):
        if not os.path.exists(settings.plot_save_folder):
            os.mkdir(settings.plot_save_folder)

    # код перемещает два файла из experiments в best указанния берет с базы данных
    # да это полный пиздец 
    def save_best_test_model(self, model: TestModel):
        best_folder = ""
        
        if not os.path.exists(settings.best_nn_folder):
            len_items = len(os.listdir(settings.best_nn_folder)) + 1
            best_folder = f"{settings.best_nn_folder}{len_items}"
            os.mkdir(best_folder)

        if os.path.exists(best_folder):
            if os.path.exists(f"{best_folder}/{os.path.basename(model.nn_model_path)}"):
                shutil.copyfile(
                    model.nn_model_path, f"{best_folder}/{os.path.basename(model.nn_model_path)}")
            if os.path.exists(f"{best_folder}/{os.path.basename(model.result_plot_path)}"):
                shutil.copyfile(
                    model.result_plot_path, f"{best_folder}/{os.path.basename(model.result_plot_path)}")

    def remove_experiments_folder(self):
        exp = glob.glob("./out/experiments/*")
        for epoch in exp:
            shutil.rmtree(epoch)

    def save_nn(self, epoch: int, running_loss: float, net: SimpleCNN) -> TrainingModel:

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")

        folder_path = f'./out/experiments/experiment_{epoch +1}_{running_loss / 2000}_{current_time}'
        os.makedirs(folder_path)
        torch.save(net.state_dict(),
                   f'{folder_path}/{settings.pre_trained_modelname}')

        return
