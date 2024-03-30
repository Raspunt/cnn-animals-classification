

import os

from sklearn.model_selection import StratifiedShuffleSplit
from settings import settings


class DatasetReader:

    def read_datasets(self) -> tuple:
        dataset_raw = os.listdir(settings.dataset_folder_path)
        dataset = []
        labels = []

        class_to_idx = {ds_class: i for i, ds_class in enumerate(dataset_raw)}

        for ds_class in dataset_raw:
            folder_path = f"{settings.dataset_folder_path}/{ds_class}"

            for img in os.listdir(folder_path):
                dataset.append(f"{folder_path}/{img}")
                labels.append(class_to_idx[ds_class])

        return (dataset, labels, class_to_idx)

    def test_train_dataset(self, dataset, labels):
        split = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(dataset, labels):
            train_dataset = [dataset[i] for i in train_index]
            test_dataset = [dataset[i] for i in test_index]

        return (train_dataset, test_dataset)
