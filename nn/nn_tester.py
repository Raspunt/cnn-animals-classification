import json
import uuid
from pathlib import Path

from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from db.crud.crud_test_model import CrudTestModel
from db.models import TestModel
from nn import device
from nn.dataset_reader import AnimalDataset
from nn.evaluation import evaluate
from nn.nn_file_manager import NNFileManager
from settings import settings


class NNTester:
    def test(self, model_path=None):
        manager = NNFileManager()
        model_path = (
            Path(model_path) if model_path is not None else manager.best_model_path
        )
        model, checkpoint = manager.load_nn(model_path)
        classes = checkpoint["classes"]
        test_files = checkpoint.get("splits", {}).get("test")
        if not test_files:
            raise ValueError(
                "The checkpoint contains no held-out test split. Retrain the model."
            )
        paths = [Path(settings.dataset_folder_path) / path for path in test_files]
        loader = DataLoader(
            AnimalDataset(paths, {name: i for i, name in enumerate(classes)}),
            batch_size=settings.batch_size,
            num_workers=settings.num_workers,
        )
        result = evaluate(model, loader, device, len(classes))
        totals = dict(zip(classes, result.confusion_matrix.sum(1).tolist()))
        correct = dict(zip(classes, result.confusion_matrix.diag().tolist()))
        predicted = dict(zip(classes, result.confusion_matrix.sum(0).tolist()))
        class_accuracies = {
            name: 100 * correct[name] / total if total else None
            for name, total in totals.items()
        }
        print(f"Using model: {model_path}; epoch: {checkpoint['epoch']}")
        for name in classes:
            accuracy = class_accuracies[name]
            value = f"{accuracy:.2f}%" if accuracy is not None else "N/A"
            print(
                f"{name}: {correct[name]}/{totals[name]} correct ({value}); predicted {predicted[name]} times"
            )
        print(f"Accuracy on test dataset: {result.accuracy:.2f}%")
        print("Confusion matrix: rows=true class, columns=predicted class")
        print(f"Class order: {classes}")
        print(result.confusion_matrix.numpy())
        manager.create_plot_folder()
        stem = Path(settings.plot_save_folder) / f"test_{uuid.uuid4().hex}"
        figure = Figure(figsize=(10, 6))
        ax = figure.subplots()
        ax.bar(classes, [class_accuracies[name] or 0 for name in classes])
        ax.set(xlabel="Animal class", ylabel="Accuracy (%)", ylim=(0, 100))
        ax.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        figure.savefig(stem.with_suffix(".png"))
        report = {
            "model_path": str(model_path),
            "epoch": checkpoint["epoch"],
            "accuracy": result.accuracy,
            "loss": result.loss,
            "classes": classes,
            "total_predictions": totals,
            "correct_predictions": correct,
            "predicted_counts": predicted,
            "class_accuracies": class_accuracies,
            "confusion_matrix": result.confusion_matrix.tolist(),
        }
        stem.with_suffix(".json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        test_model = TestModel(
            accuracy=result.accuracy,
            total_predictions=totals,
            correct_predictions=correct,
            nn_model_path=str(model_path),
            result_plot_path=str(stem.with_suffix(".png")),
        )
        CrudTestModel.create_test(test_model)
        return test_model
