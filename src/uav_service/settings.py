from pydantic import Field
from pydantic_settings import BaseSettings as _BaseSettings


class BaseSettings(
    _BaseSettings,
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
):
    """Dotenv-aware settings."""


class MiscSettings(BaseSettings):
    title: str = "Planing service"
    environment: str
    base_api_path: str = "/api/uav-service"


class Settings(BaseSettings):
    misc: MiscSettings = Field(default_factory=MiscSettings)


settings = Settings()
