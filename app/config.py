from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_provider: Literal["anthropic", "openai"] = Field(
        "anthropic",
        alias="LLM_PROVIDER",
    )
    anthropic_api_key: str | None = Field(None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field("claude-opus-4-5", alias="ANTHROPIC_MODEL")
    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-5.1", alias="OPENAI_MODEL")
    analysis_passes: int = Field(3, ge=1, le=5, alias="ANALYSIS_PASSES")
    api_key: str = Field(..., alias="API_KEY")
    port: int = Field(8000, alias="PORT")
    environment: str = Field("development", alias="ENVIRONMENT")
    app_version: str = "1.0.0"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_provider_keys(self) -> "Settings":
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
