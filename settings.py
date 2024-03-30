from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    dataset_folder_path: str = "./data/animals-10/raw-img"
    trained_model_path: str = "./out/trained.pth"
    epoch_count: int = 10
    remove_epoch_dialog: bool = False

    sql_url:str = 'sqlite:///data/Statics.db'
    plot_save_folder:str = './out/plots'


settings = Settings()
