import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from commands.runner import run
from db import session
from db.models import TestModel as EvaluationRecord
from db.models import TrainingModel
from nn import device
from nn.dataset_reader import AnimalDataset, DatasetReader
from nn.nn_file_manager import NNFileManager
from nn.nn_predicer import predict
from nn.nn_tester import NNTester
from nn.simple_cnn import SimpleCNN
from settings import settings


def constant_model():
    model = SimpleCNN(num_classes=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.fc3.bias.copy_(torch.tensor([4.0, 0.0]))
    return model


def test_labels_and_splits_are_stable(image_dataset):
    reader = DatasetReader()
    paths, labels = reader.read_datasets()
    assert reader.classes == ["butterfly", "dog"]
    assert len(paths) == 40
    assert all(
        reader.classes[label] == Path(path).parent.name
        for path, label in zip(paths, labels)
    )
    train, test = reader.test_train_dataset(paths, labels)
    assert (train, test) == tuple(reader.test_train_dataset(paths, labels))
    assert not set(train) & set(test)
    assert set(train) | set(test) == set(paths)
    assert {Path(path).parent.name for path in test} == set(reader.classes)


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_images_always_have_three_channels(tmp_path, mode):
    path = tmp_path / "dog" / "image.png"
    path.parent.mkdir()
    Image.new(mode, (50, 40)).save(path)
    image, label = AnimalDataset([path], {"dog": 1})[0]
    assert image.shape == (3, 32, 32)
    assert label == 1


def test_rgb_color_is_preserved(tmp_path):
    path = tmp_path / "dog" / "red.png"
    path.parent.mkdir()
    Image.new("RGB", (32, 32), (255, 0, 0)).save(path)
    image, _ = AnimalDataset([path], {"dog": 0})[0]
    assert image[0].mean() == 1
    assert image[1:].sum() == 0


@pytest.mark.parametrize("batched", [False, True])
def test_prediction_uses_saved_order_and_softmax(tmp_path, batched):
    manager = NNFileManager()
    # Deliberately nonalphabetical. A confident dog must never become a butterfly.
    classes = ["dog", "butterfly"]
    path = manager.save_nn(constant_model(), classes, 1, {}, {}, tmp_path / "run")
    model, checkpoint = manager.load_nn(path)
    model.train()
    image = torch.zeros(1, 3, 32, 32) if batched else torch.zeros(3, 32, 32)
    result = predict(image, checkpoint["classes"], model)
    assert result.label == "dog"
    assert result.confidence == pytest.approx(0.98201376)
    assert sum(result.probabilities.values()) == pytest.approx(1)
    assert not model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_legacy_weights_are_not_silently_mislabeled(tmp_path):
    path = tmp_path / "old.pth"
    torch.save(SimpleCNN().state_dict(), path)
    with pytest.raises(ValueError, match="no class mapping"):
        NNFileManager().load_nn(path)


def test_incompatible_preprocessing_is_rejected(tmp_path):
    manager = NNFileManager()
    path = manager.save_nn(constant_model(), ["dog", "butterfly"], 1, {}, {}, tmp_path)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["preprocessing"] = "old-grayscale"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="preprocessing"):
        manager.load_nn(path)


def test_confidence_is_not_test_accuracy(image_dataset, tmp_path, monkeypatch):
    manager = NNFileManager()
    classes = ["dog", "butterfly"]
    test_files = [f"{name}/{index:02d}.png" for name in classes for index in range(2)]
    path = manager.save_nn(
        constant_model(), classes, 1, {}, {"test": test_files}, tmp_path / "run"
    )
    # Testing must use the persisted split, even if current split settings change.
    monkeypatch.setattr(settings, "seed", 123)
    monkeypatch.setattr(settings, "test_size", 0.5)
    monkeypatch.setattr(settings, "batch_size", 3)
    result = NNTester().test(path)
    assert result.accuracy == 50.0  # The model is 98% confident on EVERY image.
    assert result.total_predictions == {"dog": 2, "butterfly": 2}
    assert result.correct_predictions == {"dog": 2, "butterfly": 0}
    report = json.loads(Path(result.result_plot_path).with_suffix(".json").read_text())
    assert report["classes"] == classes
    assert report["confusion_matrix"] == [[2, 0], [2, 0]]
    assert report["predicted_counts"] == {"dog": 4, "butterfly": 0}
    assert report["loss"] == pytest.approx(2.0181499)


def test_training_test_and_prediction_commands(image_dataset, capsys):
    torch.manual_seed(settings.seed)
    initial_model = SimpleCNN(num_classes=2)
    run(["train"])
    manager = NNFileManager()
    model, best = manager.load_nn()
    assert any(
        not torch.equal(value, initial_model.state_dict()[name])
        for name, value in model.cpu().state_dict().items()
    )
    splits = [set(best["splits"][name]) for name in ("train", "validation", "test")]
    assert len(set.union(*splits)) == 40
    assert sum(map(len, splits)) == 40
    assert all(splits)
    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in Path(settings.experiments_folder).rglob("*.pth")
    ]
    assert len(checkpoints) == 2
    expected = max(
        checkpoints,
        key=lambda c: (
            c["metrics"]["validation_accuracy"],
            -c["metrics"]["validation_loss"],
        ),
    )
    assert best["epoch"] == expected["epoch"]
    assert session.query(TrainingModel).count() == 1
    run(["test"])
    assert session.query(EvaluationRecord).count() == 1
    assert sum(session.query(EvaluationRecord).one().total_predictions.values()) == len(
        splits[2]
    )
    run(["predict", str(image_dataset / "dog" / "00.png")])
    assert "Model confidence (softmax):" in capsys.readouterr().out
    model.to(device)


def test_clear_training_command_does_not_clear_tests(monkeypatch):
    from commands.cmd_views import CmdViews

    calls = []
    monkeypatch.setattr(
        CmdViews, "clear_train_data", lambda self: calls.append("train")
    )
    monkeypatch.setattr(CmdViews, "clear_test_data", lambda self: calls.append("test"))
    run(["clear-traning-data"])
    run(["clear-training-data"])
    assert calls == ["train", "train"]


def test_missing_dataset_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "dataset_folder_path", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="one subfolder per animal class"):
        DatasetReader().read_datasets()


def test_split_rejects_missing_minority_class():
    paths = [f"image_{index}.png" for index in range(100)]
    with pytest.raises(ValueError, match="keeping every class represented"):
        DatasetReader.split(paths, [0] * 98 + [1] * 2, 0.2)
