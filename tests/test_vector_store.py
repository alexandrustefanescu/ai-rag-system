"""Tests for the vector_store module."""

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from rag_system import vector_store as vs
from rag_system.config import VectorStoreConfig
from rag_system.models import Chunk


@pytest.fixture
def tmp_db_config() -> VectorStoreConfig:
    """Return a config pointing to a temporary ChromaDB directory."""
    tmpdir = tempfile.mkdtemp()
    return VectorStoreConfig(db_path=tmpdir, collection_name="test_collection")


@pytest.fixture
def collection(tmp_db_config: VectorStoreConfig):
    client = vs.get_client(tmp_db_config)
    return vs.get_or_create_collection(client, tmp_db_config)


class TestGetClient:
    def test_returns_persistent_client(self, tmp_db_config: VectorStoreConfig) -> None:
        client = vs.get_client(tmp_db_config)
        assert client is not None

    def test_default_config(self) -> None:
        client = vs.get_client()
        assert client is not None


class TestGetOrCreateCollection:
    def test_creates_collection(self, tmp_db_config: VectorStoreConfig) -> None:
        client = vs.get_client(tmp_db_config)
        collection = vs.get_or_create_collection(client, tmp_db_config)
        assert collection is not None
        assert collection.name == "test_collection"

    def test_returns_existing_collection(
        self,
        tmp_db_config: VectorStoreConfig,
    ) -> None:
        client = vs.get_client(tmp_db_config)
        c1 = vs.get_or_create_collection(client, tmp_db_config)
        c2 = vs.get_or_create_collection(client, tmp_db_config)
        assert c1.name == c2.name


class TestAddChunks:
    def test_adds_chunks_to_collection(self, collection, sample_chunks) -> None:
        added = vs.add_chunks(collection, sample_chunks)
        assert added == 2
        assert collection.count() == 2

    def test_empty_chunks_returns_zero(self, collection) -> None:
        added = vs.add_chunks(collection, [])
        assert added == 0
        assert collection.count() == 0

    def test_batch_processing(self, collection) -> None:
        chunks = [
            Chunk(text=f"Chunk number {i}", metadata={"source": "test.txt"})
            for i in range(250)
        ]
        added = vs.add_chunks(collection, chunks, batch_size=100)
        assert added == 250
        assert collection.count() == 250


class TestQuery:
    def test_returns_results(self, collection, sample_chunks) -> None:
        vs.add_chunks(collection, sample_chunks)
        results = vs.query(collection, "Python programming", n_results=2)
        assert "documents" in results
        assert len(results["documents"][0]) <= 2

    def test_returns_distances(self, collection, sample_chunks) -> None:
        vs.add_chunks(collection, sample_chunks)
        results = vs.query(collection, "Python", n_results=1)
        assert "distances" in results
        assert len(results["distances"][0]) == 1


class TestResetCollection:
    def test_clears_existing_data(
        self,
        tmp_db_config: VectorStoreConfig,
        sample_chunks,
    ) -> None:
        client = vs.get_client(tmp_db_config)
        collection = vs.get_or_create_collection(client, tmp_db_config)
        vs.add_chunks(collection, sample_chunks)
        assert collection.count() == 2

        new_collection = vs.reset_collection(client, tmp_db_config)
        assert new_collection.count() == 0


class TestGetEmbeddingFunction:
    def test_returns_cached_function(self) -> None:
        """Test that embedding function is cached."""
        ef1 = vs.get_embedding_function("all-MiniLM-L6-v2")
        ef2 = vs.get_embedding_function("all-MiniLM-L6-v2")
        assert ef1 is ef2

    @patch(
        "rag_system.vector_store.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_different_models_different_functions(self, mock_ef_cls) -> None:
        """Test that different models return different functions."""
        mock_ef_cls.side_effect = lambda model_name: MagicMock(name=model_name)
        vs._embedding_fn_cache.pop("test-model-a", None)
        vs._embedding_fn_cache.pop("test-model-b", None)
        ef1 = vs.get_embedding_function("test-model-a")
        ef2 = vs.get_embedding_function("test-model-b")
        assert ef1 is not ef2
        vs._embedding_fn_cache.pop("test-model-a", None)
        vs._embedding_fn_cache.pop("test-model-b", None)


class TestAddChunksEdgeCases:
    def test_adds_chunks_with_sequential_ids(self, collection) -> None:
        """Test that chunk IDs are sequential."""
        chunks = [
            Chunk(text=f"Chunk {i}", metadata={"source": "test.txt"}) for i in range(5)
        ]
        vs.add_chunks(collection, chunks)

        result = collection.get()
        ids = result["ids"]
        assert ids == ["chunk_0", "chunk_1", "chunk_2", "chunk_3", "chunk_4"]

    def test_second_batch_continues_ids(self, collection, sample_chunks) -> None:
        """Test that adding more chunks continues the ID sequence."""
        vs.add_chunks(collection, sample_chunks)
        assert collection.count() == 2

        more_chunks = [Chunk(text="New chunk", metadata={"source": "new.txt"})]
        vs.add_chunks(collection, more_chunks)

        result = collection.get()
        assert "chunk_2" in result["ids"]

    def test_large_batch_size(self, collection) -> None:
        """Test with batch size larger than number of chunks."""
        chunks = [Chunk(text="Text", metadata={"source": "test.txt"}) for _ in range(5)]
        added = vs.add_chunks(collection, chunks, batch_size=100)
        assert added == 5


class TestQueryEdgeCases:
    def test_query_with_n_results_larger_than_collection(
        self,
        collection,
        sample_chunks,
    ) -> None:
        """Test querying for more results than available."""
        vs.add_chunks(collection, sample_chunks)
        results = vs.query(collection, "test", n_results=100)
        # Should return at most the number of chunks in collection
        assert len(results["documents"][0]) <= 2

    def test_query_empty_collection(self, collection) -> None:
        """Test querying an empty collection."""
        results = vs.query(collection, "test query", n_results=5)
        assert results["documents"][0] == []


class TestResetCollectionEdgeCases:
    def test_reset_nonexistent_collection(
        self, tmp_db_config: VectorStoreConfig
    ) -> None:
        """Test resetting a collection that doesn't exist yet."""
        client = vs.get_client(tmp_db_config)
        # Should not raise an error
        collection = vs.reset_collection(client, tmp_db_config)
        assert collection.count() == 0

    def test_reset_preserves_collection_name(
        self,
        tmp_db_config: VectorStoreConfig,
    ) -> None:
        """Test that reset preserves the collection name."""
        client = vs.get_client(tmp_db_config)
        coll1 = vs.reset_collection(client, tmp_db_config)
        assert coll1.name == tmp_db_config.collection_name

        coll2 = vs.reset_collection(client, tmp_db_config)
        assert coll2.name == tmp_db_config.collection_name


class TestVectorStoreWithRealData:
    def test_end_to_end_similarity_search(self, collection) -> None:
        """Test that semantically similar queries return relevant results."""
        chunks = [
            Chunk(text="Python is a programming language", metadata={"source": "1"}),
            Chunk(text="Dogs are pets that bark", metadata={"source": "2"}),
            Chunk(
                text="JavaScript is used for web development", metadata={"source": "3"}
            ),
        ]
        vs.add_chunks(collection, chunks)

        results = vs.query(collection, "coding languages", n_results=3)
        docs = results["documents"][0]

        # Programming-related docs should rank higher
        assert any("Python" in doc or "JavaScript" in doc for doc in docs[:2])
