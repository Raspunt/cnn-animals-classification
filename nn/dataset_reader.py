from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from nn import transform
from settings import settings


class AnimalDataset(Dataset):
    def __init__(self, paths, class_to_idx):
        self.paths = list(paths)
        self.class_to_idx = class_to_idx
        unknown = {Path(path).parent.name for path in self.paths} - class_to_idx.keys()
        if unknown:
            raise ValueError(f"Classes absent from the model: {sorted(unknown)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = Path(self.paths[index])
        with Image.open(path) as image:
            tensor = transform(image.convert("RGB"))
        return tensor, self.class_to_idx[path.parent.name]


class DatasetReader:
    def read_datasets(self) -> tuple:
        folder = Path(settings.dataset_folder_path)
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Dataset not found: {folder}. Expected one subfolder per animal class."
            )
        dataset = ImageFolder(folder)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        if len(self.classes) < 2:
            raise ValueError("At least two animal classes are required.")
        paths, labels = map(list, zip(*dataset.samples))
        if any(
            Path(path).parent.parent.resolve() != folder.resolve() for path in paths
        ):
            raise ValueError("Place images directly inside their class folders.")
        return paths, labels

    def test_train_dataset(self, dataset, labels):
        return self.split(dataset, labels, settings.test_size)

    @staticmethod
    def split(paths, labels, test_size):
        try:
            train_paths, test_paths, train_labels, test_labels = train_test_split(
                paths,
                labels,
                test_size=test_size,
                stratify=labels,
                random_state=settings.seed,
            )
            if set(train_labels) != set(labels) or set(test_labels) != set(labels):
                raise ValueError(
                    "A split would contain no images of at least one class."
                )
            return train_paths, test_paths
        except ValueError as error:
            raise ValueError(
                "Cannot split the dataset while keeping every class represented. "
                "Add images per class or adjust TEST_SIZE / VALIDATION_SIZE. "
                f"Details: {error}"
            ) from error
