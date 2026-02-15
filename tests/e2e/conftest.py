"""E2E test fixtures — live FastAPI server with mocked Ollama."""

import socket
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import uvicorn


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def _mock_ollama():
    """Patch ollama.chat and ollama.list for the entire test session."""
    mock_chat = MagicMock(
        return_value={"message": {"content": "This is a mocked answer from the LLM."}}
    )
    mock_list = MagicMock(
        return_value={
            "models": [
                {"name": "gemma3:1b"},
                {"name": "llama3.2:1b"},
            ]
        }
    )

    with (
        patch("ollama.chat", mock_chat),
        patch("ollama.list", mock_list),
        patch("rag_system.rag_engine.ollama.chat", mock_chat),
    ):
        yield mock_chat


@pytest.fixture(scope="session")
def _tmp_dirs():
    """Create temporary documents and ChromaDB directories."""
    with (
        tempfile.TemporaryDirectory() as docs_dir,
        tempfile.TemporaryDirectory() as db_dir,
    ):
        yield docs_dir, db_dir


@pytest.fixture(scope="session")
def live_server(_mock_ollama, _tmp_dirs):
    """Start a real uvicorn server in a background thread."""
    docs_dir, db_dir = _tmp_dirs
    port = _find_free_port()

    # Override config via env vars before importing the app
    env_patches = {
        "DOCUMENTS_DIR": docs_dir,
        "VS_DB_PATH": db_dir,
    }

    with patch.dict("os.environ", env_patches):
        # Force re-creation of config with new env vars
        import rag_system.web as web_module
        from rag_system.config import AppConfig

        web_module._config = AppConfig()

        config = uvicorn.Config(
            app=web_module.app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )
        server = uvicorn.Server(config)

        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        # Wait for server to be ready (sentence-transformers model load can be slow)
        for _ in range(300):
            try:
                sock = socket.create_connection(("127.0.0.1", port), timeout=0.2)
                sock.close()
                break
            except OSError:
                time.sleep(0.2)
        else:
            pytest.fail("Live server did not start in time")

        yield f"http://127.0.0.1:{port}"

        server.should_exit = True
        thread.join(timeout=5)


@pytest.fixture(scope="session")
def base_url(live_server):
    return live_server


@pytest.fixture
def docs_dir(_tmp_dirs):
    """Return the temporary documents directory path."""
    return Path(_tmp_dirs[0])
