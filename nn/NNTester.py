
from matplotlib import pyplot as plt
import torch

from nn.NNpredicer import predict
from nn.SimpleCNN import SimpleCNN
from settings import settings
from PIL import Image

from nn import transform, device


class NNTester():

    def __init__(self, cnn: SimpleCNN) -> None:
        self.cnn = cnn

    def test(self, test_dataset):
        self.cnn.load_state_dict(torch.load(settings.trained_model_path))

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

        print(f"total predictions:", class_total_predictions)
        print(f"correct predictions:", class_correct_predictions)
        print(
            f"Accuracy on test dataset: {100 * sum(class_correct_predictions.values()) / sum(class_total_predictions.values())} %")

        plt.bar(class_accuracies.keys(), class_accuracies.values())
        plt.xlabel('Animal Class')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy for Each Animal Class')
        plt.show()
