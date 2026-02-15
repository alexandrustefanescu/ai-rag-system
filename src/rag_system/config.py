"""Centralized configuration for the RAG system."""

import json

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkConfig(BaseSettings):
    """Text chunking parameters."""

    model_config = SettingsConfigDict(env_prefix="CHUNK_", frozen=True)

    size: int = Field(default=500, gt=0)
    overlap: int = Field(default=100, ge=0)

    @model_validator(mode="after")
    def _overlap_less_than_size(self) -> "ChunkConfig":
        if self.overlap >= self.size:
            msg = f"overlap ({self.overlap}) must be less than size ({self.size})"
            raise ValueError(msg)
        return self


class VectorStoreConfig(BaseSettings):
    """ChromaDB vector store settings."""

    model_config = SettingsConfigDict(env_prefix="VS_", frozen=True)

    db_path: str = "./chroma_db"
    collection_name: str = "rag_documents"
    embedding_model: str = "all-MiniLM-L6-v2"
    query_results: int = Field(default=5, gt=0)
    batch_size: int = Field(default=100, gt=0)


class LLMConfig(BaseSettings):
    """Ollama LLM settings."""

    model_config = SettingsConfigDict(env_prefix="LLM_", frozen=True)

    model: str = "gemma3:1b"
    available_models: list[str] = Field(
        default=["gemma3:1b", "llama3.2:1b"],
    )

    @field_validator("available_models", mode="before")
    @classmethod
    def _parse_available_models(cls, v: object) -> list[str]:
        """Accept a JSON array string or comma-separated string from env vars."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                parsed = [item.strip() for item in v.split(",") if item.strip()]
            if not isinstance(parsed, list):
                return [str(parsed)]
            return [str(item) for item in parsed]
        return v  # type: ignore[return-value]

    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0)


class SSLConfig(BaseSettings):
    """SSL/TLS settings for HTTPS."""

    model_config = SettingsConfigDict(env_prefix="SSL_", frozen=True)

    enabled: bool = True
    certfile: str = "./certs/cert.pem"
    keyfile: str = "./certs/key.pem"
    port: int = Field(default=8443, gt=0, le=65535)


class AppConfig(BaseSettings):
    """Top-level application configuration."""

    model_config = SettingsConfigDict(frozen=True)

    documents_dir: str = "./documents"
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ssl: SSLConfig = Field(default_factory=SSLConfig)
