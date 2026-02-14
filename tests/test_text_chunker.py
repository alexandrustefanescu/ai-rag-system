"""Tests for the text_chunker module."""

from rag_system.config import ChunkConfig
from rag_system.models import Document
from rag_system.text_chunker import chunk_documents, chunk_text


class TestChunkText:
    def test_short_text_returns_single_chunk(self) -> None:
        text = "Short text."
        chunks = chunk_text(text, chunk_size=500)
        assert chunks == ["Short text."]

    def test_empty_text_returns_empty_list(self) -> None:
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_text_exactly_chunk_size(self) -> None:
        text = "a" * 500
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_produces_multiple_chunks(self) -> None:
        text = "Word. " * 200  # ~1200 chars
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1

    def test_chunks_have_overlap(self) -> None:
        text = "A" * 1000
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=100)
        assert len(chunks) >= 2
        # The end of chunk 0 should overlap with the start of chunk 1
        end_of_first = chunks[0][-100:]
        start_of_second = chunks[1][:100]
        assert end_of_first == start_of_second

    def test_no_empty_chunks(self) -> None:
        text = "Hello world. " * 100
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert chunk.strip()

    def test_paragraph_boundary_preferred(self) -> None:
        text = (
            "First paragraph content here.\n\n"
            "Second paragraph content here.\n\n"
            "Third paragraph."
        )
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=5)
        assert len(chunks) >= 2


class TestChunkDocuments:
    def test_chunks_single_document(self) -> None:
        doc = Document(
            content="Short document content.",
            metadata={"source": "test.txt"},
        )
        chunks = chunk_documents([doc])
        assert len(chunks) == 1
        assert chunks[0].text == "Short document content."
        assert chunks[0].metadata["source"] == "test.txt"
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].metadata["total_chunks"] == 1

    def test_chunks_multiple_documents(self, sample_documents) -> None:
        chunks = chunk_documents(sample_documents)
        assert len(chunks) == 2
        sources = {c.metadata["source"] for c in chunks}
        assert sources == {"doc1.txt", "doc2.txt"}

    def test_custom_chunk_config(self) -> None:
        doc = Document(content="x" * 1000, metadata={"source": "big.txt"})
        config = ChunkConfig(size=200, overlap=20)
        chunks = chunk_documents([doc], config)
        assert len(chunks) > 1

    def test_empty_documents_list(self) -> None:
        chunks = chunk_documents([])
        assert chunks == []

    def test_metadata_preserved(self) -> None:
        doc = Document(
            content="Content here.",
            metadata={"source": "file.md", "file_type": ".md"},
        )
        chunks = chunk_documents([doc])
        assert chunks[0].metadata["source"] == "file.md"
        assert chunks[0].metadata["file_type"] == ".md"
        assert "chunk_index" in chunks[0].metadata
        assert "total_chunks" in chunks[0].metadata

    def test_chunk_indices_are_sequential(self) -> None:
        doc = Document(content="word " * 500, metadata={"source": "big.txt"})
        config = ChunkConfig(size=100, overlap=10)
        chunks = chunk_documents([doc], config)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))
