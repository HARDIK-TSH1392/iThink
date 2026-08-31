from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):

    # For local development
    database_url: str = "sqlite+aiosqlite:///./ithink_dev.db"

    gemini_api_key: str = ""

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()