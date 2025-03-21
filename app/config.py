from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""

    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_api_base: str = Field(default="https://api.openai.com/v1", env="OPENAI_API_BASE")
    openai_api_type: str = Field(default="open_ai", env="OPENAI_API_TYPE")
    openai_api_version: str = Field(default="v1", env="OPENAI_API_VERSION")
    openai_organization: str = Field(default="", env="OPENAI_ORGANIZATION")

    # Anthropic
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")

    # Logging
    logging_level: str = Field(default="INFO", env="LOGGING_LEVEL")

    # Miscellaneous
    workspace_root: Path = Field(default=Path(__file__).parent.parent / "workspace")
    workspace_path: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        env_nested_delimiter = "__"


config = Settings() 