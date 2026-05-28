"""Domain models for the RAG system."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """A loaded document with its text content and metadata."""

    content: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A text chunk produced from a document."""

    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedContext:
    """A single retrieved context passage with relevance score."""

    text: str
    source: str
    relevance: float


@dataclass(frozen=True)
class GenerationMetrics:
    """Performance metrics captured during LLM generation."""

    duration_s: float
    tokens_generated: int
    tokens_per_second: float


@dataclass(frozen=True)
class RAGResponse:
    """The final response from the RAG pipeline."""

    answer: str
    contexts: list[RetrievedContext] = field(default_factory=list)
    metrics: GenerationMetrics | None = None
