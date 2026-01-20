from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM
    llm_provider: str = Field(
        default="chatgpt",
        validation_alias=AliasChoices("LLM", "llm"),
    )
    gemini_model: str = Field(default="gemini-2.5-pro", env="GEMINI_MODEL")
    openai_model: str = Field(default="gpt-5.2", env="OPENAI_MODEL")

    # Embeddings / cache
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="portfolio-agent-cache", env="PINECONE_INDEX_NAME")

    # External tools
    tavily_api_key: Optional[str] = Field(default=None, env="TAVILY_API_KEY")
    e2b_api_key: Optional[str] = Field(default=None, env="E2B_API_KEY")

    # Robinhood
    robinhood_username: Optional[str] = Field(default=None, env="ROBINHOOD_USERNAME")
    robinhood_password: Optional[str] = Field(default=None, env="ROBINHOOD_PASSWORD")
    robinhood_totp: Optional[str] = Field(default=None, env="ROBINHOOD_TOTP")
    robinhood_use_live: bool = Field(default=False, env="ROBINHOOD_USE_LIVE")
    robinhood_csv_path: Optional[str] = Field(default=None, env="ROBINHOOD_CSV_PATH")

    # Paths
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3] / "data")
    log_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3] / "logs")

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[3] / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Ensure .env is loaded once for the process from project root
    env_path = Path(__file__).resolve().parents[3] / ".env"
    # override=True lets .env values win over any pre-set shell envs
    load_dotenv(env_path, override=True)
    return Settings()
