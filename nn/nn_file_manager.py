import shutil
from pathlib import Path

import torch

from nn import PREPROCESSING, device
from nn.simple_cnn import SimpleCNN
from settings import settings


class NNFileManager:
    @property
    def best_model_path(self) -> Path:
        return Path(settings.best_nn_folder) / settings.pre_trained_modelname

    def create_plot_folder(self):
        Path(settings.plot_save_folder).mkdir(parents=True, exist_ok=True)

    def remove_experiments_folder(self):
        folder = Path(settings.experiments_folder)
        if folder.exists():
            shutil.rmtree(folder)
        # The best model is preserved; clearing experiments does not delete it.

    def save_nn(self, net, classes, epoch, metrics, splits, run_folder) -> Path:
        path = Path(run_folder) / f"epoch_{epoch:03d}" / settings.pre_trained_modelname
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "model_state_dict": net.state_dict(),
                "classes": list(classes),
                "preprocessing": PREPROCESSING,
                "epoch": epoch,
                "metrics": metrics,
                "splits": splits,
                "seed": settings.seed,
            },
            path,
        )
        return path

    def save_best_model(self, model_path) -> Path:
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, self.best_model_path)
        return self.best_model_path

    def load_nn(self, model_path=None):
        path = Path(model_path) if model_path is not None else self.best_model_path
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}. Run 'train' first.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "classes" not in checkpoint:
            raise ValueError(
                "This legacy checkpoint has no class mapping. Its animal labels "
                "cannot be recovered reliably from the weights. Retrain the model."
            )
        classes = checkpoint["classes"]
        if (
            not isinstance(classes, list)
            or len(classes) < 2
            or not all(isinstance(name, str) and name for name in classes)
            or len(set(classes)) != len(classes)
        ):
            raise ValueError("Invalid class mapping in checkpoint.")
        if (
            checkpoint.get("format_version") != 1
            or checkpoint.get("preprocessing") != PREPROCESSING
        ):
            raise ValueError(
                "Incompatible model format or image preprocessing. Retrain the model."
            )
        model = SimpleCNN(num_classes=len(classes))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        return model, checkpoint
