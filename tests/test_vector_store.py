"""Tests for the vector_store module."""

import tempfile

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
