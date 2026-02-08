"""Tests for the __main__ module."""

from unittest.mock import patch


class TestMainEntryPoint:
    @patch("rag_system.cli.main")
    def test_calls_main(self, mock_main) -> None:
        with open("src/rag_system/__main__.py") as f:
            code = f.read()

        # Execute the module code with main patched
        exec(compile(code, "__main__.py", "exec"), {"__name__": "__test__"})
        mock_main.assert_called_once()
