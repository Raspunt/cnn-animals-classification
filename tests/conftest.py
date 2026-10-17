import os

os.environ["SQL_URL"] = "sqlite:///:memory:"
os.environ["MPLBACKEND"] = "Agg"

import pytest  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from db import Base, engine, init_db, session  # noqa: E402
from settings import settings  # noqa: E402

torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def isolated_outputs(tmp_path, monkeypatch):
    for setting, directory in [
        ("experiments_folder", "experiments"),
        ("best_nn_folder", "best"),
        ("plot_save_folder", "plots"),
    ]:
        monkeypatch.setattr(settings, setting, str(tmp_path / directory))
    monkeypatch.setattr(settings, "seed", 42)
    monkeypatch.setattr(settings, "epoch_count", 2)
    monkeypatch.setattr(settings, "batch_size", 4)
    monkeypatch.setattr(settings, "num_workers", 0)
    monkeypatch.setattr(settings, "test_size", 0.2)
    monkeypatch.setattr(settings, "validation_size", 0.2)
    monkeypatch.setattr(settings, "learning_rate", 0.01)
    monkeypatch.setattr(settings, "momentum", 0.9)
    session.rollback()
    Base.metadata.drop_all(engine)
    init_db()
    yield
    session.rollback()
    session.close()


@pytest.fixture
def image_dataset(tmp_path, monkeypatch):
    root = tmp_path / "arbitrary" / "nested" / "animals"
    # Deliberately create the folders in the opposite order from their class IDs.
    for name, color in [("dog", (220, 20, 20)), ("butterfly", (20, 20, 220))]:
        folder = root / name
        folder.mkdir(parents=True)
        for index in range(20):
            Image.new("RGB", (40, 40), color).save(folder / f"{index:02d}.png")
        (folder / "notes.txt").write_text("not an image", encoding="utf-8")
    monkeypatch.setattr(settings, "dataset_folder_path", str(root))
    return root
