"""Tests for configuration module."""

import pytest
from pydantic import ValidationError

from rag_system.config import (
    AppConfig,
    ChunkConfig,
    LLMConfig,
    SSLConfig,
    VectorStoreConfig,
)


class TestChunkConfig:
    def test_defaults(self) -> None:
        c = ChunkConfig()
        assert c.size == 500
        assert c.overlap == 100

    def test_custom_values(self) -> None:
        c = ChunkConfig(size=1000, overlap=100)
        assert c.size == 1000
        assert c.overlap == 100

    def test_is_frozen(self) -> None:
        c = ChunkConfig()
        with pytest.raises(ValidationError):
            c.size = 999

    def test_rejects_negative_size(self) -> None:
        with pytest.raises(ValidationError):
            ChunkConfig(size=-1)

    def test_rejects_zero_size(self) -> None:
        with pytest.raises(ValidationError):
            ChunkConfig(size=0)

    def test_rejects_negative_overlap(self) -> None:
        with pytest.raises(ValidationError):
            ChunkConfig(overlap=-1)

    def test_rejects_overlap_equal_to_size(self) -> None:
        with pytest.raises(ValidationError):
            ChunkConfig(size=100, overlap=100)

    def test_rejects_overlap_greater_than_size(self) -> None:
        with pytest.raises(ValidationError):
            ChunkConfig(size=100, overlap=200)

    def test_allows_zero_overlap(self) -> None:
        c = ChunkConfig(overlap=0)
        assert c.overlap == 0


class TestVectorStoreConfig:
    def test_defaults(self) -> None:
        c = VectorStoreConfig()
        assert c.db_path == "./chroma_db"
        assert c.collection_name == "rag_documents"
        assert c.embedding_model == "all-MiniLM-L6-v2"
        assert c.query_results == 5
        assert c.batch_size == 100

    def test_rejects_zero_query_results(self) -> None:
        with pytest.raises(ValidationError):
            VectorStoreConfig(query_results=0)

    def test_rejects_zero_batch_size(self) -> None:
        with pytest.raises(ValidationError):
            VectorStoreConfig(batch_size=0)


class TestLLMConfig:
    def test_defaults(self) -> None:
        c = LLMConfig()
        assert c.model == "gemma3:1b"
        assert c.temperature == 0.3
        assert c.max_tokens == 512

    def test_rejects_negative_temperature(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfig(temperature=-0.1)

    def test_rejects_temperature_above_max(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfig(temperature=2.1)

    def test_allows_boundary_temperatures(self) -> None:
        assert LLMConfig(temperature=0.0).temperature == 0.0
        assert LLMConfig(temperature=2.0).temperature == 2.0

    def test_rejects_zero_max_tokens(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=0)

    def test_available_models_from_json_string(self) -> None:
        c = LLMConfig(available_models='["a", "b"]')
        assert c.available_models == ["a", "b"]

    def test_available_models_from_csv_string(self) -> None:
        c = LLMConfig(available_models="gemma3:1b, llama3.2:1b")
        assert c.available_models == ["gemma3:1b", "llama3.2:1b"]

    def test_available_models_from_list(self) -> None:
        c = LLMConfig(available_models=["x", "y"])
        assert c.available_models == ["x", "y"]

    def test_available_models_csv_strips_whitespace(self) -> None:
        c = LLMConfig(available_models=" a , b , c ")
        assert c.available_models == ["a", "b", "c"]

    def test_available_models_from_single_json_value(self) -> None:
        """When JSON parses to a non-list (e.g. bare string), wrap in list."""
        c = LLMConfig(available_models='"single-model"')
        assert c.available_models == ["single-model"]


class TestAppConfig:
    def test_defaults(self) -> None:
        c = AppConfig()
        assert c.documents_dir == "./documents"
        assert isinstance(c.chunk, ChunkConfig)
        assert isinstance(c.vector_store, VectorStoreConfig)
        assert isinstance(c.llm, LLMConfig)

    def test_custom_nested(self) -> None:
        c = AppConfig(
            llm=LLMConfig(model="llama3"),
            chunk=ChunkConfig(size=1000),
        )
        assert c.llm.model == "llama3"
        assert c.chunk.size == 1000

    def test_includes_ssl_config(self) -> None:
        c = AppConfig()
        assert isinstance(c.ssl, SSLConfig)
        assert c.ssl.enabled is True

    def test_is_frozen(self) -> None:
        c = AppConfig()
        with pytest.raises(ValidationError):
            c.documents_dir = "/new/path"

    def test_custom_documents_dir(self) -> None:
        c = AppConfig(documents_dir="/custom/docs")
        assert c.documents_dir == "/custom/docs"


class TestSSLConfigValidation:
    def test_allows_valid_port_range(self) -> None:
        c1 = SSLConfig(port=1)
        assert c1.port == 1

        c2 = SSLConfig(port=65535)
        assert c2.port == 65535

        c3 = SSLConfig(port=443)
        assert c3.port == 443

    def test_custom_cert_paths(self) -> None:
        c = SSLConfig(certfile="/custom/cert.pem", keyfile="/custom/key.pem")
        assert c.certfile == "/custom/cert.pem"
        assert c.keyfile == "/custom/key.pem"

    def test_ssl_can_be_disabled(self) -> None:
        c = SSLConfig(enabled=False)
        assert c.enabled is False


class TestLLMConfigValidation:
    def test_available_models_default(self) -> None:
        c = LLMConfig()
        assert "gemma3:1b" in c.available_models
        assert "llama3.2:1b" in c.available_models

    def test_custom_available_models(self) -> None:
        c = LLMConfig(available_models=["model1", "model2"])
        assert c.available_models == ["model1", "model2"]

    def test_accepts_fractional_temperature(self) -> None:
        c = LLMConfig(temperature=0.7)
        assert c.temperature == 0.7

    def test_accepts_large_max_tokens(self) -> None:
        c = LLMConfig(max_tokens=4096)
        assert c.max_tokens == 4096

    def test_rejects_negative_max_tokens(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=-1)


class TestVectorStoreConfigValidation:
    def test_custom_embedding_model(self) -> None:
        c = VectorStoreConfig(embedding_model="custom-model-v2")
        assert c.embedding_model == "custom-model-v2"

    def test_custom_collection_name(self) -> None:
        c = VectorStoreConfig(collection_name="my_collection")
        assert c.collection_name == "my_collection"

    def test_custom_db_path(self) -> None:
        c = VectorStoreConfig(db_path="/data/chroma")
        assert c.db_path == "/data/chroma"

    def test_rejects_negative_query_results(self) -> None:
        with pytest.raises(ValidationError):
            VectorStoreConfig(query_results=-1)

    def test_rejects_negative_batch_size(self) -> None:
        with pytest.raises(ValidationError):
            VectorStoreConfig(batch_size=-1)


class TestChunkConfigEdgeCases:
    def test_maximum_overlap_less_than_size(self) -> None:
        """Test that overlap can be size - 1."""
        c = ChunkConfig(size=100, overlap=99)
        assert c.overlap == 99

    def test_large_chunk_size(self) -> None:
        c = ChunkConfig(size=10000, overlap=1000)
        assert c.size == 10000
        assert c.overlap == 1000

    def test_minimal_chunk_size(self) -> None:
        c = ChunkConfig(size=1, overlap=0)
        assert c.size == 1
        assert c.overlap == 0
