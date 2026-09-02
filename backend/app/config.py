from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):

    # For local development
    database_url: str = "sqlite+aiosqlite:///./ithink_dev.db"

    gemini_api_key: str = ""

    # From Agora Console -> Project -> notification config. Empty until a
    # webhook is actually registered there (needs a public HTTPS URL, so
    # this stays unset for local dev). See iCall_utils.verify_agora_signature.
    agora_webhook_secret: str = ""

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()