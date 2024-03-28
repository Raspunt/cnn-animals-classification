from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    dataset_folder_path: str = "./data/animals-10/raw-img"
    trained_model_path: str = "./out/trained.pth"
    epoch_count:int = 5


settings = Settings()