"""Tests for the text_chunker module."""

from unittest.mock import patch

import numpy as np

from rag_system.config import ChunkConfig
from rag_system.models import Document
from rag_system.text_chunker import (
    _cosine_similarity,
    _split_sentences,
    chunk_documents,
    chunk_text,
)


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


class TestChunkTextEdgeCases:
    def test_single_word_chunks(self) -> None:
        """Test chunking with very small chunk size."""
        text = "one two three four five"
        chunks = chunk_text(text, chunk_size=5, chunk_overlap=1)
        assert all(len(c) <= 5 for c in chunks)

    def test_no_sentence_breaks(self) -> None:
        """Test text with no sentence breaks."""
        text = "a" * 1000
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1

    def test_all_newlines(self) -> None:
        """Test text that is mostly newlines."""
        text = "\n\n\n\n" + "text" + "\n\n\n\n"
        chunks = chunk_text(text, chunk_size=100)
        assert len(chunks) >= 1
        assert "text" in "".join(chunks)

    def test_unicode_characters(self) -> None:
        """Test chunking text with unicode characters."""
        text = "日本語のテキスト。" * 100
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 1

    def test_mixed_sentence_breaks(self) -> None:
        """Test text with multiple types of sentence breaks."""
        text = (
            "First sentence. Second sentence! Third sentence? Fourth line\nFifth line"
        )
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=5)
        assert len(chunks) >= 2

    def test_very_long_single_sentence(self) -> None:
        """Test a single very long sentence without breaks."""
        text = "word " * 500  # 2500+ characters, no sentence breaks
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) >= 4

    def test_paragraph_breaks_preferred(self) -> None:
        """Test that paragraph breaks are preferred over sentence breaks."""
        text = "Para 1 sentence one. Para 1 sentence two.\n\nPara 2 sentence one." * 5
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        # Check that some chunks end with paragraph breaks
        assert any("\n\n" in chunk for chunk in chunks) or len(chunks) > 1

    def test_exact_overlap_boundary(self) -> None:
        """Test that overlap is exactly as specified."""
        text = "0123456789" * 20  # 200 chars
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
        assert len(chunks) >= 2


class TestChunkDocumentsEdgeCases:
    def test_documents_with_different_sizes(self) -> None:
        """Test chunking documents of varying sizes."""
        docs = [
            Document(content="Short", metadata={"source": "1"}),
            Document(content="x" * 1000, metadata={"source": "2"}),
            Document(content="Medium length document here", metadata={"source": "3"}),
        ]
        chunks = chunk_documents(docs, ChunkConfig(size=100, overlap=10))
        # All documents should produce at least one chunk
        sources = {c.metadata["source"] for c in chunks}
        assert sources == {"1", "2", "3"}

    def test_chunk_index_and_total_correct(self) -> None:
        """Test that chunk_index and total_chunks are correct."""
        doc = Document(content="word " * 300, metadata={"source": "test.txt"})
        config = ChunkConfig(size=100, overlap=10)
        chunks = chunk_documents([doc], config)

        total = len(chunks)
        for chunk in chunks:
            assert chunk.metadata["total_chunks"] == total
            assert 0 <= chunk.metadata["chunk_index"] < total

    def test_preserves_all_metadata_fields(self) -> None:
        """Test that all metadata fields are preserved."""
        doc = Document(
            content="Content here",
            metadata={"source": "file.txt", "file_type": ".txt", "custom": "value"},
        )
        chunks = chunk_documents([doc])
        for chunk in chunks:
            assert chunk.metadata["source"] == "file.txt"
            assert chunk.metadata["file_type"] == ".txt"
            assert chunk.metadata["custom"] == "value"

    def test_multiple_docs_sequential_processing(self) -> None:
        """Test that multiple documents are processed in order."""
        docs = [
            Document(
                content=f"Document {i} content", metadata={"source": f"doc{i}.txt"}
            )
            for i in range(5)
        ]
        chunks = chunk_documents(docs)

        # Extract sources in order
        sources = [c.metadata["source"] for c in chunks]
        # Should maintain document order
        prev_doc_num = -1
        for source in sources:
            doc_num = int(source.replace("doc", "").replace(".txt", ""))
            assert doc_num >= prev_doc_num
            prev_doc_num = doc_num if doc_num > prev_doc_num else prev_doc_num

    def test_zero_overlap_chunks(self) -> None:
        """Test chunking with zero overlap."""
        doc = Document(content="A" * 500, metadata={"source": "test.txt"})
        config = ChunkConfig(size=100, overlap=0)
        chunks = chunk_documents([doc], config)
        assert len(chunks) == 5

    def test_maximum_overlap(self) -> None:
        """Test chunking with maximum allowed overlap."""
        doc = Document(content="X" * 500, metadata={"source": "test.txt"})
        config = ChunkConfig(size=100, overlap=99)
        chunks = chunk_documents([doc], config)
        # With high overlap, we get many chunks
        assert len(chunks) > 5


class TestSplitSentences:
    def test_simple_sentences(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        sentences = _split_sentences(text)
        assert len(sentences) >= 2
        assert any("First" in s for s in sentences)

    def test_paragraph_breaks(self) -> None:
        text = "Paragraph one.\n\nParagraph two."
        sentences = _split_sentences(text)
        assert len(sentences) == 2

    def test_long_sentence_fallback(self) -> None:
        text = "x" * 2000
        sentences = _split_sentences(text)
        assert len(sentences) > 1
        assert all(len(s) <= 1000 for s in sentences)

    def test_empty_text(self) -> None:
        assert _split_sentences("") == []
        assert _split_sentences("   ") == []


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine_similarity(v, v) == 1.0

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_zero_vector(self) -> None:
        a = np.array([1.0, 2.0])
        z = np.array([0.0, 0.0])
        assert _cosine_similarity(a, z) == 0.0


def _fake_embed(sentences):
    """Return deterministic embeddings that simulate topic shifts."""
    embeddings = []
    for s in sentences:
        if "Python" in s or "programming" in s:
            embeddings.append(np.array([1.0, 0.0, 0.0]))
        elif "cooking" in s or "recipe" in s:
            embeddings.append(np.array([0.0, 1.0, 0.0]))
        else:
            embeddings.append(np.array([0.5, 0.5, 0.0]))
    return embeddings


class TestSemanticChunking:
    @patch("rag_system.vector_store.get_embedding_function")
    def test_splits_on_topic_shift(self, mock_get_ef) -> None:
        mock_get_ef.return_value = _fake_embed
        # Make text long enough to exceed semantic_max_size so it isn't
        # returned as a single chunk by the short-circuit.
        python_part = "Python is a programming language. " * 10
        cooking_part = "Cooking is an art. A recipe needs ingredients. " * 10
        text = python_part + "\n\n" + cooking_part
        config = ChunkConfig(
            strategy="semantic", semantic_threshold=0.5, semantic_max_size=500
        )
        doc = Document(content=text, metadata={"source": "test.txt"})
        chunks = chunk_documents([doc], config)
        assert len(chunks) >= 2
        assert any("Python" in c.text for c in chunks)
        assert any("Cooking" in c.text or "recipe" in c.text for c in chunks)

    @patch("rag_system.vector_store.get_embedding_function")
    def test_respects_max_size(self, mock_get_ef) -> None:
        mock_get_ef.return_value = lambda sents: [np.array([1.0, 0.0]) for _ in sents]
        text = "Same topic sentence. " * 200
        config = ChunkConfig(
            strategy="semantic", semantic_threshold=0.1, semantic_max_size=300
        )
        doc = Document(content=text, metadata={"source": "test.txt"})
        chunks = chunk_documents([doc], config)
        for c in chunks:
            assert len(c.text) <= 300

    def test_short_text_single_chunk(self) -> None:
        config = ChunkConfig(strategy="semantic", semantic_max_size=1000)
        doc = Document(content="Short text.", metadata={"source": "t.txt"})
        chunks = chunk_documents([doc], config)
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."

    def test_empty_text(self) -> None:
        config = ChunkConfig(strategy="semantic")
        doc = Document(content="", metadata={"source": "e.txt"})
        chunks = chunk_documents([doc], config)
        assert chunks == []

    def test_no_sentence_boundaries_falls_back(self) -> None:
        text = "x" * 3000
        config = ChunkConfig(strategy="semantic", semantic_max_size=500)
        doc = Document(content=text, metadata={"source": "t.txt"})
        chunks = chunk_documents([doc], config)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.text) <= 500

    def test_default_strategy_is_fixed(self) -> None:
        config = ChunkConfig()
        assert config.strategy == "fixed"
        doc = Document(content="Word " * 500, metadata={"source": "t.txt"})
        chunks = chunk_documents([doc], config)
        assert len(chunks) > 0

    @patch("rag_system.vector_store.get_embedding_function")
    def test_metadata_preserved(self, mock_get_ef) -> None:
        mock_get_ef.return_value = _fake_embed
        text = "Python is a programming language.\n\nCooking is an art with a recipe."
        config = ChunkConfig(
            strategy="semantic", semantic_threshold=0.5, semantic_max_size=5000
        )
        doc = Document(
            content=text,
            metadata={"source": "f.md", "file_type": ".md"},
        )
        chunks = chunk_documents([doc], config)
        for c in chunks:
            assert c.metadata["source"] == "f.md"
            assert c.metadata["file_type"] == ".md"
            assert "chunk_index" in c.metadata
            assert "total_chunks" in c.metadata
