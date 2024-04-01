import os
import uuid
import json

from matplotlib import pyplot as plt
import pandas as pd
import torch

from nn.nn_predicer import predict
from settings import settings
from PIL import Image

from nn import transform, device, net
from db.models import TestModel
from db.crud.crud_test_model import CrudTestModel


class NNTester():


    def get_experiment_by_epoch(self, epoch: int):
        experiments = os.listdir(settings.experiments_folder)
        epoch_files = [exp for exp in experiments if str(
            epoch) == exp.split("_")[1]]
        if epoch_files:
            best_experiment = sorted(epoch_files)[0]
            return f"{settings.experiments_folder}/{best_experiment}/{settings.pre_trained_modelname}"
        else:
            return None

    def find_best_experiment(self, i) -> str:
        experiments = os.listdir(settings.experiments_folder)

        loss_arr = [exp.split("_")[2] for exp in experiments]
        min_loss = min(loss_arr)

        best_experiment = [exp for exp in experiments if min_loss in exp.split("_")[
            1]]

        return f"{settings.experiments_folder}/{best_experiment[0]}/{settings.pre_trained_modelname}"

        # for exp in experiments:
        #     print(exp.split('_')[2])

    def test(self, test_dataset, epoch):

        model_path = self.get_experiment_by_epoch(epoch)
        print(f"using model from:{model_path}")
        net.load_state_dict(torch.load(model_path))

        test_labels = [test_class.split('/')[4] for test_class in test_dataset]
        test_labels = list(set(test_labels))

        class_correct_predictions = {label: 0 for label in test_labels}
        class_total_predictions = {label: 0 for label in test_labels}

        for i, img_path in enumerate(test_dataset):

            img_raw = Image.open(img_path)
            img = transform(img_raw)
            img = img.to(device)

            true_label = img_path.split('/')[4]
            class_total_predictions[true_label] += 1

            if predict(img, test_labels) == true_label:
                class_correct_predictions[true_label] += 1

        class_accuracies = {class_name: (class_correct_predictions[class_name] / class_total_predictions[class_name]) if class_total_predictions[class_name] != 0 else 0
                            for class_name in test_labels}

        accuracy = 100 * sum(class_correct_predictions.values()) / \
            sum(class_total_predictions.values())


        df = pd.DataFrame([class_total_predictions, class_correct_predictions], 
                  index=['Total Predictions', 'Correct Predictions'])
        
        print(df)
        print(
            f"Accuracy on test dataset: {accuracy} %")

        test_model = TestModel()
        test_model.accuracy = accuracy
        test_model.total_predictions = json.dumps(class_total_predictions)
        test_model.correct_predictions = json.dumps(class_correct_predictions)
        test_model.nn_model_path = model_path

        plt.bar(class_accuracies.keys(), class_accuracies.values())
        plt.xlabel('Animal Class')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy for Each Animal Class')
        # plt.show()

        plot_save_path = f"{settings.plot_save_folder}/test_plot_{str(uuid.uuid4())}.png"
        plt.savefig(plot_save_path)

        test_model.result_plot_path = plot_save_path
        CrudTestModel.create_test(test_model)

