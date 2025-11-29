from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Bacterial ML App"
    MODEL_DIR: str = "app/ml_models"
    UPLOAD_DIR: str = "local_storage/uploads"
    RESULT_DIR: str = "local_storage/results"

    class Config:
        env_file = ".env"

settings = Settings()
