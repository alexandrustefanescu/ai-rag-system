"""Unit tests for the FastAPI web interface."""

import io
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rag_system.models import RAGResponse, RetrievedContext


@asynccontextmanager
async def _noop_lifespan(app):
    yield


@pytest.fixture(autouse=True)
def _tmp_config(tmp_path):
    """Patch web module config and inject mock ChromaDB into app.state.

    The lifespan is replaced with a no-op so that TestClient does not
    create a real ChromaDB client/collection on startup.
    """
    import rag_system.web as web

    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_client = MagicMock()

    original_config = web._config
    original_lifespan = web.app.router.lifespan_context

    web.app.router.lifespan_context = _noop_lifespan
    web._config = web.AppConfig(documents_dir=str(tmp_path / "docs"))
    web.app.state.chroma_client = mock_client
    web.app.state.chroma_collection = mock_collection

    (tmp_path / "docs").mkdir()

    yield web, mock_collection, mock_client

    web._config = original_config
    web.app.router.lifespan_context = original_lifespan


@pytest.fixture
def client():
    """Create a TestClient with the no-op lifespan."""
    import rag_system.web as web

    with TestClient(web.app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def docs_dir(_tmp_config):
    """Return the temp documents directory."""
    web = _tmp_config[0]
    return Path(web._config.documents_dir)


# ---------- Health endpoint ----------


class TestHealth:
    def test_healthy_when_ollama_reachable(self, client):
        with patch("ollama.list", return_value={"models": []}):
            resp = client.get("/api/v1/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["ollama_connected"] is True

    def test_degraded_when_ollama_unreachable(self, client):
        with patch("ollama.list", side_effect=Exception("connection refused")):
            resp = client.get("/api/v1/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["ollama_connected"] is False

    def test_returns_document_count(self, client, _tmp_config):
        _, collection, _ = _tmp_config
        collection.count.return_value = 42

        with patch("ollama.list"):
            resp = client.get("/api/v1/health")

        assert resp.json()["documents"] == 42


# ---------- Status endpoint ----------


class TestStatus:
    def test_returns_model_info(self, client):
        with patch("ollama.list"):
            resp = client.get("/api/v1/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "gemma3:1b"
        assert "gemma3:1b" in data["available_models"]

    def test_returns_document_count(self, client, _tmp_config):
        _, collection, _ = _tmp_config
        collection.count.return_value = 10

        with patch("ollama.list"):
            resp = client.get("/api/v1/status")

        assert resp.json()["documents"] == 10

    def test_ollama_disconnected(self, client):
        with patch("ollama.list", side_effect=RuntimeError("no ollama")):
            resp = client.get("/api/v1/status")

        assert resp.json()["ollama_connected"] is False


# ---------- Ingest endpoint ----------


class TestIngest:
    def test_no_documents_found(self, client, docs_dir):
        resp = client.post("/api/v1/ingest")

        assert resp.status_code == 200
        assert resp.json()["status"] == "no_documents"
        assert resp.json()["chunks"] == 0

    def test_successful_ingest(self, client, docs_dir, _tmp_config):
        (docs_dir / "hello.txt").write_text("Hello world for testing ingest.")

        with patch("rag_system.web.vs") as mock_vs:
            mock_vs.reset_collection.return_value = _tmp_config[1]
            mock_vs.add_chunks.return_value = 1
            resp = client.post("/api/v1/ingest")

        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["chunks"] == 1


# ---------- Upload endpoint ----------


class TestUpload:
    def _make_file(self, name: str, content: bytes):
        return ("files", (name, io.BytesIO(content), "application/octet-stream"))

    def test_upload_txt_file(self, client, docs_dir, _tmp_config):
        with patch("rag_system.web.vs") as mock_vs:
            mock_vs.reset_collection.return_value = _tmp_config[1]
            mock_vs.add_chunks.return_value = 1
            resp = client.post(
                "/api/v1/upload",
                files=[self._make_file("test.txt", b"Some test content.")],
            )

        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["files_saved"] == 1
        assert (docs_dir / "test.txt").exists()

    def test_upload_md_file(self, client, docs_dir, _tmp_config):
        with patch("rag_system.web.vs") as mock_vs:
            mock_vs.reset_collection.return_value = _tmp_config[1]
            mock_vs.add_chunks.return_value = 1
            resp = client.post(
                "/api/v1/upload",
                files=[self._make_file("notes.md", b"# Title\nContent here.")],
            )

        assert resp.json()["status"] == "ok"
        assert (docs_dir / "notes.md").exists()

    def test_rejects_unsupported_extension(self, client, docs_dir):
        resp = client.post(
            "/api/v1/upload",
            files=[self._make_file("image.png", b"\x89PNG")],
        )

        assert resp.status_code == 200
        assert resp.json()["status"] == "no_valid_files"
        assert resp.json()["files_saved"] == 0
        assert not (docs_dir / "image.png").exists()

    def test_rejects_exe_extension(self, client, docs_dir):
        resp = client.post(
            "/api/v1/upload",
            files=[self._make_file("malware.exe", b"\x00\x01")],
        )

        assert resp.json()["status"] == "no_valid_files"

    def test_rejects_oversized_file(self, client, docs_dir):
        large_content = b"x" * (51 * 1024 * 1024)
        resp = client.post(
            "/api/v1/upload",
            files=[self._make_file("big.txt", large_content)],
        )

        assert resp.json()["status"] == "no_valid_files"
        assert not (docs_dir / "big.txt").exists()

    def test_path_traversal_saves_inside_docs_dir(self, client, docs_dir, _tmp_config):
        with patch("rag_system.web.vs") as mock_vs:
            mock_vs.reset_collection.return_value = _tmp_config[1]
            mock_vs.add_chunks.return_value = 1
            resp = client.post(
                "/api/v1/upload",
                files=[self._make_file("../../etc/passwd.txt", b"hacked")],
            )

        # Basename extraction sanitizes to "passwd.txt" — saved inside docs_dir
        assert resp.json()["files_saved"] == 1
        assert (docs_dir / "passwd.txt").exists()
        assert not (docs_dir.parent.parent / "etc" / "passwd.txt").exists()

    def test_multiple_files(self, client, docs_dir, _tmp_config):
        with patch("rag_system.web.vs") as mock_vs:
            mock_vs.reset_collection.return_value = _tmp_config[1]
            mock_vs.add_chunks.return_value = 2
            resp = client.post(
                "/api/v1/upload",
                files=[
                    self._make_file("a.txt", b"File A"),
                    self._make_file("b.txt", b"File B"),
                ],
            )

        assert resp.json()["files_saved"] == 2
        assert (docs_dir / "a.txt").exists()
        assert (docs_dir / "b.txt").exists()

    def test_mixed_valid_and_invalid(self, client, docs_dir, _tmp_config):
        with patch("rag_system.web.vs") as mock_vs:
            mock_vs.reset_collection.return_value = _tmp_config[1]
            mock_vs.add_chunks.return_value = 1
            resp = client.post(
                "/api/v1/upload",
                files=[
                    self._make_file("valid.txt", b"OK"),
                    self._make_file("bad.exe", b"NOPE"),
                ],
            )

        assert resp.json()["files_saved"] == 1
        assert (docs_dir / "valid.txt").exists()
        assert not (docs_dir / "bad.exe").exists()


# ---------- Document list endpoint ----------


class TestListDocuments:
    def test_empty_collection(self, client, _tmp_config):
        _, collection, _ = _tmp_config
        collection.count.return_value = 0

        resp = client.get("/api/v1/documents")

        assert resp.status_code == 200
        assert resp.json()["files"] == []

    def test_lists_documents_from_vector_store(self, client, _tmp_config, docs_dir):
        _, collection, _ = _tmp_config
        collection.count.return_value = 3
        collection.get.return_value = {
            "metadatas": [
                {"source": "file1.txt"},
                {"source": "file1.txt"},
                {"source": "file2.md"},
            ],
        }

        (docs_dir / "file1.txt").write_text("Hello world")

        resp = client.get("/api/v1/documents")

        data = resp.json()
        assert len(data["files"]) == 2
        file_names = [f["filename"] for f in data["files"]]
        assert "file1.txt" in file_names
        assert "file2.md" in file_names

        f1 = next(f for f in data["files"] if f["filename"] == "file1.txt")
        assert f1["chunk_count"] == 2

        f2 = next(f for f in data["files"] if f["filename"] == "file2.md")
        assert f2["chunk_count"] == 1
        assert f2["size_kb"] == 0

    def test_file_on_disk_has_size(self, client, _tmp_config):
        """When a file exists in docs_dir, size_kb is populated."""
        web, collection, _ = _tmp_config
        docs = Path(web._config.documents_dir)
        (docs / "sized.txt").write_text("x" * 2048)
        collection.count.return_value = 1
        collection.get.return_value = {"metadatas": [{"source": "sized.txt"}]}

        resp = client.get("/api/v1/documents")

        f = resp.json()["files"][0]
        assert f["size_kb"] > 0

    def test_handles_missing_metadata(self, client, _tmp_config):
        _, collection, _ = _tmp_config
        collection.count.return_value = 1
        collection.get.return_value = {"metadatas": [None]}

        resp = client.get("/api/v1/documents")

        data = resp.json()
        assert len(data["files"]) == 1
        assert data["files"][0]["filename"] == "unknown"

    def test_handles_empty_source_key(self, client, _tmp_config):
        _, collection, _ = _tmp_config
        collection.count.return_value = 1
        collection.get.return_value = {"metadatas": [{}]}

        resp = client.get("/api/v1/documents")

        assert resp.json()["files"][0]["filename"] == "unknown"


# ---------- Delete document endpoint ----------


class TestDeleteDocument:
    def test_delete_existing_file(self, client, docs_dir, _tmp_config):
        _, collection, _ = _tmp_config
        (docs_dir / "to_delete.txt").write_text("Delete me")
        collection.count.return_value = 1
        collection.get.return_value = {
            "ids": ["chunk_1"],
            "metadatas": [{"source": "to_delete.txt"}],
        }

        resp = client.delete("/api/v1/documents/to_delete.txt")

        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert not (docs_dir / "to_delete.txt").exists()
        collection.delete.assert_called_once_with(ids=["chunk_1"])

    def test_delete_nonexistent_file(self, client, _tmp_config):
        _, collection, _ = _tmp_config
        collection.count.return_value = 0

        resp = client.delete("/api/v1/documents/ghost.txt")

        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_delete_removes_only_matching_chunks(self, client, docs_dir, _tmp_config):
        _, collection, _ = _tmp_config
        collection.count.return_value = 3
        collection.get.return_value = {
            "ids": ["c1", "c2", "c3"],
            "metadatas": [
                {"source": "target.txt"},
                {"source": "other.txt"},
                {"source": "target.txt"},
            ],
        }

        resp = client.delete("/api/v1/documents/target.txt")

        collection.delete.assert_called_once_with(ids=["c1", "c3"])

    def test_delete_file_only_on_disk_no_chunks(self, client, docs_dir, _tmp_config):
        _, collection, _ = _tmp_config
        (docs_dir / "orphan.txt").write_text("No chunks for this")
        collection.count.return_value = 2
        collection.get.return_value = {
            "ids": ["c1", "c2"],
            "metadatas": [
                {"source": "other1.txt"},
                {"source": "other2.txt"},
            ],
        }

        resp = client.delete("/api/v1/documents/orphan.txt")

        assert resp.json()["status"] == "ok"
        assert not (docs_dir / "orphan.txt").exists()
        collection.delete.assert_not_called()

    def test_delete_returns_remaining_chunk_count(self, client, _tmp_config):
        _, collection, _ = _tmp_config
        collection.count.return_value = 5
        collection.get.return_value = {
            "ids": ["c1"],
            "metadatas": [{"source": "x.txt"}],
        }

        resp = client.delete("/api/v1/documents/x.txt")

        assert resp.json()["chunks"] == 5


# ---------- Ask endpoint ----------


class TestAsk:
    def test_ask_returns_answer_and_sources(self, client):
        mock_response = RAGResponse(
            answer="Python is great.",
            contexts=[
                RetrievedContext(text="Python info", source="docs.txt", relevance=0.95),
            ],
        )

        with patch("rag_system.web.rag_engine.ask", return_value=mock_response):
            resp = client.post(
                "/api/v1/ask",
                json={"question": "What is Python?"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Python is great."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["source"] == "docs.txt"
        assert data["sources"][0]["relevance"] == 0.95

    def test_ask_with_model_override(self, client):
        mock_response = RAGResponse(answer="Answer", contexts=[])

        with patch("rag_system.web.rag_engine.ask", return_value=mock_response) as mock_ask:
            resp = client.post(
                "/api/v1/ask",
                json={"question": "Hello?", "model": "llama3.2:1b"},
            )

        assert resp.status_code == 200
        call_kwargs = mock_ask.call_args
        config_used = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config_used.model == "llama3.2:1b"

    def test_ask_with_invalid_model_uses_default(self, client):
        mock_response = RAGResponse(answer="Default model answer", contexts=[])

        with patch("rag_system.web.rag_engine.ask", return_value=mock_response) as mock_ask:
            resp = client.post(
                "/api/v1/ask",
                json={"question": "Hello?", "model": "nonexistent-model"},
            )

        assert resp.status_code == 200
        call_kwargs = mock_ask.call_args
        config_used = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config_used.model == "gemma3:1b"

    def test_ask_without_model_uses_default(self, client):
        mock_response = RAGResponse(answer="Answer", contexts=[])

        with patch("rag_system.web.rag_engine.ask", return_value=mock_response) as mock_ask:
            resp = client.post(
                "/api/v1/ask",
                json={"question": "Tell me something"},
            )

        assert resp.status_code == 200
        call_kwargs = mock_ask.call_args
        config_used = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config_used.model == "gemma3:1b"

    def test_ask_with_empty_question(self, client):
        mock_response = RAGResponse(answer="No relevant documents found.", contexts=[])

        with patch("rag_system.web.rag_engine.ask", return_value=mock_response):
            resp = client.post("/api/v1/ask", json={"question": ""})

        assert resp.status_code == 200

    def test_ask_missing_question_field(self, client):
        resp = client.post("/api/v1/ask", json={})

        assert resp.status_code == 422

    def test_ask_with_multiple_sources(self, client):
        mock_response = RAGResponse(
            answer="Combined answer.",
            contexts=[
                RetrievedContext(text="Source 1 text", source="a.txt", relevance=0.9),
                RetrievedContext(text="Source 2 text", source="b.md", relevance=0.8),
                RetrievedContext(text="Source 3 text", source="c.txt", relevance=0.7),
            ],
        )

        with patch("rag_system.web.rag_engine.ask", return_value=mock_response):
            resp = client.post(
                "/api/v1/ask",
                json={"question": "Multi-source question"},
            )

        assert len(resp.json()["sources"]) == 3
        sources = [s["source"] for s in resp.json()["sources"]]
        assert sources == ["a.txt", "b.md", "c.txt"]

    def test_ask_no_body(self, client):
        resp = client.post("/api/v1/ask")

        assert resp.status_code == 422


# ---------- SSLConfig ----------


class TestSSLConfig:
    def test_defaults(self):
        from rag_system.config import SSLConfig

        c = SSLConfig()
        assert c.enabled is True
        assert c.certfile == "./certs/cert.pem"
        assert c.keyfile == "./certs/key.pem"
        assert c.port == 8443

    def test_is_frozen(self):
        from pydantic import ValidationError
        from rag_system.config import SSLConfig

        c = SSLConfig()
        with pytest.raises(ValidationError):
            c.enabled = False

    def test_rejects_invalid_port(self):
        from pydantic import ValidationError
        from rag_system.config import SSLConfig

        with pytest.raises(ValidationError):
            SSLConfig(port=0)
        with pytest.raises(ValidationError):
            SSLConfig(port=70000)
