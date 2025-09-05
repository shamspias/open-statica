from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    APP_NAME: str = "OpenStatica"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    API_PREFIX: str = "/api"

    # Security
    SECRET_KEY: str = "your-secret-key-here"

    # Storage
    UPLOAD_PATH: str = "./uploads"
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: List[str] = [".csv", ".xlsx", ".xls"]

    class Config:
        env_file = ".env"


def get_settings() -> Settings:
    return Settings()
