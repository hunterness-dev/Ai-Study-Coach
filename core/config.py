from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AI-Study-Coach"
    app_env: str = "development"
    log_level: str = "INFO"

    database_url: str = "sqlite:///./data/study_coach.db"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    daily_study_hours: float = 8.0
    rl_model_path: str = "models/ppo_study_agent"
    ml_model_path: str = "models/ridge_pipeline.joblib"


@lru_cache
def get_settings() -> Settings:
    return Settings()
