"""Tests for the cli module."""

from unittest.mock import MagicMock, patch

import pytest

from rag_system.cli import _setup_logging, chat, ingest, main
from rag_system.config import AppConfig
from rag_system.models import Document, RAGResponse


class TestSetupLogging:
    def test_default_level_is_info(self) -> None:
        with patch("rag_system.cli.logging.basicConfig") as mock_basic:
            _setup_logging()
            mock_basic.assert_called_once()
            assert mock_basic.call_args[1]["level"] == 20  # logging.INFO

    def test_verbose_sets_debug(self) -> None:
        with patch("rag_system.cli.logging.basicConfig") as mock_basic:
            _setup_logging(verbose=True)
            mock_basic.assert_called_once()
            assert mock_basic.call_args[1]["level"] == 10  # logging.DEBUG


class TestIngest:
    @patch("rag_system.cli.vs")
    @patch("rag_system.cli.chunk_documents")
    @patch("rag_system.cli.load_documents")
    def test_successful_ingest(self, mock_load, mock_chunk, mock_vs) -> None:
        mock_load.return_value = [
            Document(content="doc1", metadata={"source": "a.txt"}),
        ]
        mock_chunk.return_value = [MagicMock(), MagicMock()]
        mock_vs.add_chunks.return_value = 2

        ingest("/some/folder")

        mock_load.assert_called_once_with("/some/folder")
        mock_chunk.assert_called_once()
        mock_vs.get_client.assert_called_once()
        mock_vs.reset_collection.assert_called_once()
        mock_vs.add_chunks.assert_called_once()

    @patch("rag_system.cli.load_documents")
    def test_no_documents_found(self, mock_load, capsys) -> None:
        mock_load.return_value = []

        ingest("/empty/folder")

        captured = capsys.readouterr()
        assert "No supported documents found" in captured.out

    @patch("rag_system.cli.vs")
    @patch("rag_system.cli.chunk_documents")
    @patch("rag_system.cli.load_documents")
    def test_uses_provided_config(self, mock_load, mock_chunk, mock_vs) -> None:
        mock_load.return_value = [
            Document(content="doc1", metadata={"source": "a.txt"}),
        ]
        mock_chunk.return_value = [MagicMock()]
        mock_vs.add_chunks.return_value = 1
        cfg = AppConfig()

        ingest("/some/folder", config=cfg)

        mock_chunk.assert_called_once_with(mock_load.return_value, cfg.chunk)


class TestChat:
    @patch("rag_system.cli.vs")
    def test_empty_collection_exits(self, mock_vs, capsys) -> None:
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_vs.get_or_create_collection.return_value = mock_collection

        chat()

        captured = capsys.readouterr()
        assert "No documents in the vector store" in captured.out

    @patch("builtins.input", side_effect=["quit"])
    @patch("rag_system.cli.vs")
    def test_quit_command(self, mock_vs, mock_input, capsys) -> None:
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_vs.get_or_create_collection.return_value = mock_collection

        chat()

        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out

    @patch("builtins.input", side_effect=["exit"])
    @patch("rag_system.cli.vs")
    def test_exit_command(self, mock_vs, mock_input, capsys) -> None:
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_vs.get_or_create_collection.return_value = mock_collection

        chat()

        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out

    @patch("builtins.input", side_effect=["q"])
    @patch("rag_system.cli.vs")
    def test_q_command(self, mock_vs, mock_input, capsys) -> None:
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_vs.get_or_create_collection.return_value = mock_collection

        chat()

        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out

    @patch("builtins.input", side_effect=["", "quit"])
    @patch("rag_system.cli.vs")
    def test_empty_input_skipped(self, mock_vs, mock_input) -> None:
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_vs.get_or_create_collection.return_value = mock_collection

        chat()

        # Should have called input twice (empty + quit)
        assert mock_input.call_count == 2

    @patch("rag_system.cli.rag_engine")
    @patch("builtins.input", side_effect=["What is Python?", "quit"])
    @patch("rag_system.cli.vs")
    def test_asks_question_and_prints_answer(
        self, mock_vs, mock_input, mock_engine, capsys
    ) -> None:
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_vs.get_or_create_collection.return_value = mock_collection
        mock_engine.ask.return_value = RAGResponse(answer="Python is a language.")

        chat()

        mock_engine.ask.assert_called_once()
        captured = capsys.readouterr()
        assert "Python is a language." in captured.out

    @patch("builtins.input", side_effect=EOFError)
    @patch("rag_system.cli.vs")
    def test_eof_exits_gracefully(self, mock_vs, mock_input, capsys) -> None:
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_vs.get_or_create_collection.return_value = mock_collection

        chat()

        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    @patch("rag_system.cli.vs")
    def test_keyboard_interrupt_exits_gracefully(
        self, mock_vs, mock_input, capsys
    ) -> None:
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_vs.get_or_create_collection.return_value = mock_collection

        chat()

        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out


class TestMain:
    @patch("rag_system.cli.ingest")
    @patch("rag_system.cli._setup_logging")
    def test_ingest_command(self, mock_logging, mock_ingest) -> None:
        with patch("sys.argv", ["rag_system", "ingest", "--folder", "/docs"]):
            main()

        mock_logging.assert_called_once_with(False)
        mock_ingest.assert_called_once_with("/docs")

    @patch("rag_system.cli.chat")
    @patch("rag_system.cli._setup_logging")
    def test_chat_command(self, mock_logging, mock_chat) -> None:
        with patch("sys.argv", ["rag_system", "chat", "--model", "llama3"]):
            main()

        mock_logging.assert_called_once_with(False)
        mock_chat.assert_called_once()
        cfg = mock_chat.call_args[0][0]
        assert cfg.llm.model == "llama3"

    @patch("rag_system.cli._setup_logging")
    def test_no_command_exits_with_error(self, mock_logging) -> None:
        with patch("sys.argv", ["rag_system"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch("rag_system.cli.ingest")
    @patch("rag_system.cli._setup_logging")
    def test_verbose_flag(self, mock_logging, mock_ingest) -> None:
        with patch("sys.argv", ["rag_system", "-v", "ingest"]):
            main()

        mock_logging.assert_called_once_with(True)

    @patch("rag_system.cli.ingest")
    @patch("rag_system.cli._setup_logging")
    def test_ingest_default_folder(self, mock_logging, mock_ingest) -> None:
        with patch("sys.argv", ["rag_system", "ingest"]):
            main()

        mock_ingest.assert_called_once_with("./documents")

    @patch("rag_system.cli.chat")
    @patch("rag_system.cli._setup_logging")
    def test_chat_default_model(self, mock_logging, mock_chat) -> None:
        with patch("sys.argv", ["rag_system", "chat"]):
            main()

        cfg = mock_chat.call_args[0][0]
        assert cfg.llm.model == "tinyllama"
