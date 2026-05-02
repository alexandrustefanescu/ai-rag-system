"""Shared fixtures for the test suite."""

import tempfile
from pathlib import Path

import pytest

from rag_system.models import Chunk, Document


@pytest.fixture
def sample_text() -> str:
    return (
        "Python is a high-level programming language. "
        "It was created by Guido van Rossum. "
        "Python supports multiple paradigms. "
        "It is widely used in data science and web development."
    )


@pytest.fixture
def sample_document(sample_text: str) -> Document:
    return Document(
        content=sample_text,
        metadata={"source": "test.txt", "file_type": ".txt"},
    )


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            content="First document about Python programming.",
            metadata={"source": "doc1.txt", "file_type": ".txt"},
        ),
        Document(
            content="Second document about machine learning concepts.",
            metadata={"source": "doc2.txt", "file_type": ".txt"},
        ),
    ]


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            text="Python is a programming language.",
            metadata={"source": "test.txt", "chunk_index": 0, "total_chunks": 2},
        ),
        Chunk(
            text="It supports multiple paradigms.",
            metadata={"source": "test.txt", "chunk_index": 1, "total_chunks": 2},
        ),
    ]


@pytest.fixture
def tmp_docs_dir() -> Path:
    """Create a temporary directory with sample documents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)

        (d / "sample.txt").write_text(
            "This is a sample text document for testing the RAG system.",
            encoding="utf-8",
        )
        (d / "notes.md").write_text(
            "# Notes\n\nThis is a **markdown** document with some content.",
            encoding="utf-8",
        )
        # Unsupported file — should be skipped.
        (d / "image.png").write_bytes(b"\x89PNG\r\n")

        yield d


@pytest.fixture
def empty_docs_dir() -> Path:
    """Create an empty temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
