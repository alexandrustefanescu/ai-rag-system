"""Centralized configuration for the RAG system."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChunkConfig(BaseModel):
    """Text chunking parameters."""

    model_config = ConfigDict(frozen=True)

    size: int = Field(default=500, gt=0)
    overlap: int = Field(default=50, ge=0)

    @model_validator(mode="after")
    def _overlap_less_than_size(self) -> "ChunkConfig":
        if self.overlap >= self.size:
            msg = f"overlap ({self.overlap}) must be less than size ({self.size})"
            raise ValueError(msg)
        return self


class VectorStoreConfig(BaseModel):
    """ChromaDB vector store settings."""

    model_config = ConfigDict(frozen=True)

    db_path: str = "./chroma_db"
    collection_name: str = "rag_documents"
    embedding_model: str = "all-MiniLM-L6-v2"
    query_results: int = Field(default=3, gt=0)
    batch_size: int = Field(default=100, gt=0)


class LLMConfig(BaseModel):
    """Ollama LLM settings."""

    model_config = ConfigDict(frozen=True)

    model: str = "tinyllama"
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0)


class AppConfig(BaseModel):
    """Top-level application configuration."""

    model_config = ConfigDict(frozen=True)

    documents_dir: str = "./documents"
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
