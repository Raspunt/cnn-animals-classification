import os

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

POSTGRESS_USER = os.getenv('POSTGRESS_USER')
POSTGRESS_PASSWORD = os.getenv('POSTGRESS_PASSWORD')
POSTGRESS_HOST = os.getenv('POSTGRESS_HOST', 'localhost')
POSTGRESS_PORT = os.getenv('POSTGRESS_PORT', '5432')
POSTGRES_DB = os.getenv('POSTGRES_DB')


class Settings(BaseSettings):
    dataset_folder_path: str = "./data/animals-10/raw-img"
    epoch_count: int = 10

    best_nn_folder:str = "./out/best/"
    plot_save_folder: str = './out/plots'
    experiments_folder: str = "./out/experiments"
    pre_trained_modelname: str = "trained.pth"

    sql_url: str = f"postgresql://{POSTGRESS_USER}:{POSTGRESS_PASSWORD}@{POSTGRESS_HOST}:{POSTGRESS_PORT}/{POSTGRES_DB}"


settings = Settings()
