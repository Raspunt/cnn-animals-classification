import os

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL

load_dotenv()


def default_sql_url() -> str:
    # Keep existing PostgreSQL installations working; new checkouts use SQLite.
    if os.getenv("POSTGRESS_USER") and os.getenv("POSTGRES_DB"):
        return URL.create(
            "postgresql",
            username=os.environ["POSTGRESS_USER"],
            password=os.getenv("POSTGRESS_PASSWORD"),
            host=os.getenv("POSTGRESS_HOST", "localhost"),
            port=int(os.getenv("POSTGRESS_PORT", "5432")),
            database=os.environ["POSTGRES_DB"],
        ).render_as_string(hide_password=False)
    return "sqlite:///./out/experiments.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    dataset_folder_path: str = "./data/animals-10/raw-img"
    epoch_count: int = Field(default=10, gt=0)
    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=0.01, gt=0)
    momentum: float = Field(default=0.9, ge=0, lt=1)
    seed: int = 42
    test_size: float = Field(default=0.2, gt=0, lt=1)
    validation_size: float = Field(default=0.2, gt=0, lt=1)
    num_workers: int = Field(default=0, ge=0)

    best_nn_folder: str = "./out/best"
    plot_save_folder: str = "./out/plots"
    experiments_folder: str = "./out/experiments"
    pre_trained_modelname: str = "trained.pth"
    sql_url: str = Field(default_factory=default_sql_url)


settings = Settings()
