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
    """
    Get an available TCP port assigned by the operating system.
    
    Returns:
        port (int): An available TCP port number assigned by the OS.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def _mock_ollama():
    """
    Patch Ollama's chat and list calls used by the application and provide the chat mock to tests.
    
    This fixture replaces `ollama.chat` and `ollama.list` (including usage inside rag_system.rag_engine) with MagicMock objects that simulate a chat response and a model list for the duration of the test session.
    
    Returns:
        MagicMock: The mock used for `ollama.chat`, configured to return a predictable chat response.
    """
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
    """
    Provide temporary directories for documents and ChromaDB for use by tests.
    
    Returns:
        tuple[str, str]: A tuple (docs_dir, db_dir) containing filesystem paths to the temporary
        documents directory and the temporary ChromaDB directory. Both directories are removed
        when the fixture context exits.
    """
    with (
        tempfile.TemporaryDirectory() as docs_dir,
        tempfile.TemporaryDirectory() as db_dir,
    ):
        yield docs_dir, db_dir


@pytest.fixture(scope="session")
def live_server(_mock_ollama, _tmp_dirs):
    """
    Provide a live FastAPI server URL backed by a real uvicorn process for tests.
    
    Starts a uvicorn server in a background thread with environment variables set to use the provided temporary document and database directories, yields the server's base URL for test use, and ensures the server is stopped after the test session.
    Returns:
        base_url (str): The server base URL, e.g. "http://127.0.0.1:<port>".
    """
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
    """
    Expose the live server's base URL to tests.
    
    Parameters:
        live_server (str): The base URL of the running test server, e.g. "http://127.0.0.1:12345".
    
    Returns:
        base_url (str): The same base URL string provided by the `live_server` fixture.
    """
    return live_server


@pytest.fixture
def docs_dir(_tmp_dirs):
    """
    Get the temporary documents directory path.
    
    Parameters:
        _tmp_dirs (tuple): A tuple where the first element is the temporary documents directory path (string).
    
    Returns:
        docs_dir (Path): A Path pointing to the temporary documents directory.
    """
    return Path(_tmp_dirs[0])