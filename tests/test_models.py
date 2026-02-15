"""Tests for the models module."""

import pytest

from rag_system.models import Chunk, Document, RAGResponse, RetrievedContext


class TestDocument:
    def test_create_document(self) -> None:
        doc = Document(content="Hello world", metadata={"source": "test.txt"})
        assert doc.content == "Hello world"
        assert doc.metadata["source"] == "test.txt"

    def test_document_is_frozen(self) -> None:
        doc = Document(content="Hello", metadata={})
        with pytest.raises(AttributeError):
            doc.content = "Changed"

    def test_document_with_empty_metadata(self) -> None:
        doc = Document(content="Content")
        assert doc.metadata == {}

    def test_document_preserves_metadata(self) -> None:
        meta = {"source": "file.txt", "file_type": ".txt", "custom": "value"}
        doc = Document(content="Text", metadata=meta)
        assert doc.metadata == meta

    def test_document_with_long_content(self) -> None:
        long_content = "word " * 10000
        doc = Document(content=long_content)
        assert len(doc.content) == len(long_content)


class TestChunk:
    def test_create_chunk(self) -> None:
        chunk = Chunk(text="Chunk text", metadata={"source": "test.txt"})
        assert chunk.text == "Chunk text"
        assert chunk.metadata["source"] == "test.txt"

    def test_chunk_is_frozen(self) -> None:
        chunk = Chunk(text="Text", metadata={})
        with pytest.raises(AttributeError):
            chunk.text = "Changed"

    def test_chunk_with_empty_metadata(self) -> None:
        chunk = Chunk(text="Text")
        assert chunk.metadata == {}

    def test_chunk_with_index_metadata(self) -> None:
        chunk = Chunk(
            text="Text",
            metadata={"source": "f.txt", "chunk_index": 0, "total_chunks": 5},
        )
        assert chunk.metadata["chunk_index"] == 0
        assert chunk.metadata["total_chunks"] == 5

    def test_chunk_equality(self) -> None:
        c1 = Chunk(text="Same", metadata={"source": "a.txt"})
        c2 = Chunk(text="Same", metadata={"source": "a.txt"})
        assert c1 == c2

    def test_chunk_inequality(self) -> None:
        c1 = Chunk(text="Text1", metadata={})
        c2 = Chunk(text="Text2", metadata={})
        assert c1 != c2


class TestRetrievedContext:
    def test_create_retrieved_context(self) -> None:
        ctx = RetrievedContext(text="Context", source="doc.txt", relevance=0.95)
        assert ctx.text == "Context"
        assert ctx.source == "doc.txt"
        assert ctx.relevance == 0.95

    def test_retrieved_context_is_frozen(self) -> None:
        ctx = RetrievedContext(text="Text", source="src", relevance=0.5)
        with pytest.raises(AttributeError):
            ctx.relevance = 0.9

    def test_relevance_boundaries(self) -> None:
        ctx1 = RetrievedContext(text="Perfect", source="a", relevance=1.0)
        assert ctx1.relevance == 1.0

        ctx2 = RetrievedContext(text="None", source="b", relevance=0.0)
        assert ctx2.relevance == 0.0

    def test_retrieved_context_equality(self) -> None:
        ctx1 = RetrievedContext(text="Same", source="a.txt", relevance=0.9)
        ctx2 = RetrievedContext(text="Same", source="a.txt", relevance=0.9)
        assert ctx1 == ctx2


class TestRAGResponse:
    def test_create_rag_response(self) -> None:
        ctx = RetrievedContext(text="Context", source="doc.txt", relevance=0.9)
        response = RAGResponse(answer="The answer", contexts=[ctx])
        assert response.answer == "The answer"
        assert len(response.contexts) == 1
        assert response.contexts[0].source == "doc.txt"

    def test_rag_response_is_frozen(self) -> None:
        response = RAGResponse(answer="Answer", contexts=[])
        with pytest.raises(AttributeError):
            response.answer = "Changed"

    def test_rag_response_with_empty_contexts(self) -> None:
        response = RAGResponse(answer="No context answer")
        assert response.contexts == []

    def test_rag_response_with_multiple_contexts(self) -> None:
        contexts = [
            RetrievedContext(text=f"Context {i}", source=f"doc{i}.txt", relevance=0.9 - i * 0.1)
            for i in range(5)
        ]
        response = RAGResponse(answer="Multi-context answer", contexts=contexts)
        assert len(response.contexts) == 5

    def test_rag_response_equality(self) -> None:
        ctx = RetrievedContext(text="Text", source="a", relevance=0.9)
        r1 = RAGResponse(answer="Same", contexts=[ctx])
        r2 = RAGResponse(answer="Same", contexts=[ctx])
        assert r1 == r2


class TestModelInteroperability:
    def test_document_to_chunk_metadata_transfer(self) -> None:
        """Test that metadata can be transferred from Document to Chunk."""
        doc = Document(content="Content", metadata={"source": "file.txt", "type": ".txt"})
        chunk = Chunk(text="Chunk from doc", metadata={**doc.metadata, "chunk_index": 0})
        assert chunk.metadata["source"] == "file.txt"
        assert chunk.metadata["type"] == ".txt"
        assert chunk.metadata["chunk_index"] == 0

    def test_contexts_list_in_rag_response(self) -> None:
        """Test that RAGResponse can hold a list of contexts."""
        contexts = [
            RetrievedContext(text="Text1", source="a.txt", relevance=0.9),
            RetrievedContext(text="Text2", source="b.txt", relevance=0.8),
        ]
        response = RAGResponse(answer="Answer", contexts=contexts)
        assert response.contexts[0].text == "Text1"
        assert response.contexts[1].text == "Text2"


class TestModelEdgeCases:
    def test_empty_content_document(self) -> None:
        """Test document with empty content."""
        doc = Document(content="")
        assert doc.content == ""

    def test_empty_text_chunk(self) -> None:
        """Test chunk with empty text."""
        chunk = Chunk(text="")
        assert chunk.text == ""

    def test_zero_relevance_context(self) -> None:
        """Test context with zero relevance."""
        ctx = RetrievedContext(text="Irrelevant", source="doc", relevance=0.0)
        assert ctx.relevance == 0.0

    def test_negative_relevance_allowed(self) -> None:
        """Test that negative relevance is allowed (no validation)."""
        ctx = RetrievedContext(text="Text", source="src", relevance=-0.1)
        assert ctx.relevance == -0.1

    def test_relevance_above_one_allowed(self) -> None:
        """Test that relevance above 1.0 is allowed (no validation)."""
        ctx = RetrievedContext(text="Text", source="src", relevance=1.5)
        assert ctx.relevance == 1.5

    def test_unicode_in_models(self) -> None:
        """Test that models handle unicode correctly."""
        doc = Document(content="日本語テキスト", metadata={"source": "日本語.txt"})
        assert "日本語" in doc.content

        chunk = Chunk(text="Émojis 🎉🎊", metadata={})
        assert "🎉" in chunk.text

        ctx = RetrievedContext(text="Ünïcödé", source="fîlé.txt", relevance=0.9)
        assert "Ü" in ctx.text
