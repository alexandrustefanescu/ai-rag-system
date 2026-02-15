"""Tests for the rag_engine module."""

from unittest.mock import MagicMock, patch

import pytest

from rag_system.config import LLMConfig
from rag_system.models import RAGResponse, RetrievedContext
from rag_system.rag_engine import (
    _build_context_string,
    _parse_results,
    ask,
    generate_answer,
)


class TestParseResults:
    def test_parses_valid_results(self) -> None:
        results = {
            "documents": [["Doc about Python", "Doc about ML"]],
            "metadatas": [[{"source": "a.txt"}, {"source": "b.txt"}]],
            "distances": [[0.1, 0.3]],
        }
        contexts = _parse_results(results)
        assert len(contexts) == 2
        assert contexts[0].source == "a.txt"
        assert contexts[0].relevance == pytest.approx(0.9, abs=0.01)
        assert contexts[1].relevance == pytest.approx(0.7, abs=0.01)

    def test_handles_empty_results(self) -> None:
        results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        contexts = _parse_results(results)
        assert contexts == []

    def test_handles_missing_keys(self) -> None:
        contexts = _parse_results({})
        assert contexts == []


class TestBuildContextString:
    def test_formats_single_context(self) -> None:
        ctx = [RetrievedContext(text="Hello world", source="test.txt", relevance=0.95)]
        result = _build_context_string(ctx)
        assert "test.txt" in result
        assert "0.95" in result
        assert "Hello world" in result

    def test_formats_multiple_contexts(self) -> None:
        ctxs = [
            RetrievedContext(text="First", source="a.txt", relevance=0.9),
            RetrievedContext(text="Second", source="b.txt", relevance=0.8),
        ]
        result = _build_context_string(ctxs)
        assert "---" in result
        assert "First" in result
        assert "Second" in result

    def test_empty_contexts(self) -> None:
        result = _build_context_string([])
        assert result == ""


class TestGenerateAnswer:
    @patch("rag_system.rag_engine.ollama")
    def test_calls_ollama_with_correct_params(self, mock_ollama) -> None:
        mock_ollama.chat.return_value = {"message": {"content": "Test answer"}}
        config = LLMConfig(model="testmodel", temperature=0.5, max_tokens=256)

        answer = generate_answer("What is Python?", "Python is a language.", config)

        assert answer == "Test answer"
        mock_ollama.chat.assert_called_once()
        call_kwargs = mock_ollama.chat.call_args
        assert call_kwargs.kwargs["model"] == "testmodel"
        assert call_kwargs.kwargs["options"]["temperature"] == 0.5

    @patch("rag_system.rag_engine.ollama")
    def test_uses_default_config(self, mock_ollama) -> None:
        mock_ollama.chat.return_value = {"message": {"content": "Answer"}}
        generate_answer("Q?", "Context.")
        call_kwargs = mock_ollama.chat.call_args
        assert call_kwargs.kwargs["model"] == "gemma3:1b"


class TestAsk:
    @patch("rag_system.rag_engine.generate_answer")
    @patch("rag_system.rag_engine.vs.query")
    def test_full_pipeline(self, mock_query, mock_generate) -> None:
        mock_query.return_value = {
            "documents": [["Python info"]],
            "metadatas": [[{"source": "docs.txt"}]],
            "distances": [[0.15]],
        }
        mock_generate.return_value = "Python is great!"

        collection = MagicMock()
        response = ask("Tell me about Python", collection)

        assert isinstance(response, RAGResponse)
        assert response.answer == "Python is great!"
        assert len(response.contexts) == 1
        assert response.contexts[0].source == "docs.txt"

    @patch("rag_system.rag_engine.vs.query")
    def test_no_results_returns_fallback(self, mock_query) -> None:
        mock_query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        collection = MagicMock()
        response = ask("Unknown topic", collection)
        assert "No relevant documents" in response.answer
        assert response.contexts == []

    @patch("rag_system.rag_engine.generate_answer")
    @patch("rag_system.rag_engine.vs.query")
    def test_filters_low_relevance_results(self, mock_query, mock_generate) -> None:
        """Test that results below the relevance threshold are filtered out."""
        mock_query.return_value = {
            "documents": [["High relevance doc", "Low relevance doc"]],
            "metadatas": [[{"source": "high.txt"}, {"source": "low.txt"}]],
            "distances": [[0.1, 0.9]],  # 0.9 distance = 0.1 relevance (below 0.3)
        }
        mock_generate.return_value = "Answer from high relevance doc"
        collection = MagicMock()
        response = ask("test query", collection)

        # Should only include high relevance result
        assert len(response.contexts) == 1
        assert response.contexts[0].source == "high.txt"
        assert "No relevant documents" not in response.answer

    @patch("rag_system.rag_engine.vs.query")
    def test_all_low_relevance_returns_fallback(self, mock_query) -> None:
        """Test fallback when all results are below threshold."""
        mock_query.return_value = {
            "documents": [["Doc 1", "Doc 2"]],
            "metadatas": [[{"source": "a.txt"}, {"source": "b.txt"}]],
            "distances": [[0.9, 0.95]],  # Both below threshold
        }
        collection = MagicMock()
        response = ask("test query", collection)

        assert "No relevant documents" in response.answer
        assert response.contexts == []

    @patch("rag_system.rag_engine.generate_answer")
    @patch("rag_system.rag_engine.vs.query")
    def test_custom_n_results(self, mock_query, mock_generate) -> None:
        """Test that n_results parameter is passed correctly."""
        mock_query.return_value = {
            "documents": [["Doc"]],
            "metadatas": [[{"source": "test.txt"}]],
            "distances": [[0.1]],
        }
        mock_generate.return_value = "Answer"

        collection = MagicMock()
        ask("query", collection, n_results=10)

        mock_query.assert_called_once()
        call_args = mock_query.call_args
        assert call_args.kwargs["n_results"] == 10 or call_args[0][2] == 10

    @patch("rag_system.rag_engine.generate_answer")
    @patch("rag_system.rag_engine.vs.query")
    def test_uses_custom_config(self, mock_query, mock_generate) -> None:
        """Test that custom LLM config is passed to generate_answer."""
        mock_query.return_value = {
            "documents": [["Doc"]],
            "metadatas": [[{"source": "test.txt"}]],
            "distances": [[0.1]],
        }
        mock_generate.return_value = "Answer"

        collection = MagicMock()
        custom_config = LLMConfig(model="custom-model")
        ask("query", collection, config=custom_config)

        mock_generate.assert_called_once()
        call_args = mock_generate.call_args
        config_arg = (
            call_args[0][2] if len(call_args[0]) > 2 else call_args.kwargs.get("config")
        )
        assert config_arg.model == "custom-model"


class TestParseResultsEdgeCases:
    def test_handles_missing_source_in_metadata(self) -> None:
        """Test that missing source defaults to 'unknown'."""
        results = {
            "documents": [["Doc text"]],
            "metadatas": [[{}]],
            "distances": [[0.2]],
        }
        contexts = _parse_results(results)
        assert len(contexts) == 1
        assert contexts[0].source == "unknown"

    def test_handles_none_metadata(self) -> None:
        """Test handling of None in metadatas list."""
        results = {
            "documents": [["Doc text"]],
            "metadatas": [[None]],
            "distances": [[0.2]],
        }
        contexts = _parse_results(results)
        assert len(contexts) == 1
        assert contexts[0].source == "unknown"

    def test_relevance_calculation_boundary(self) -> None:
        """Test relevance calculation at boundaries."""
        results = {
            "documents": [["Doc1", "Doc2", "Doc3"]],
            "metadatas": [[{"source": "a"}, {"source": "b"}, {"source": "c"}]],
            "distances": [[0.0, 0.5, 1.0]],
        }
        contexts = _parse_results(results)
        assert contexts[0].relevance == 1.0  # Perfect match
        assert contexts[1].relevance == 0.5  # Mid-range
        assert contexts[2].relevance == 0.0  # No match


class TestBuildContextStringEdgeCases:
    def test_preserves_newlines_in_text(self) -> None:
        """Test that newlines in context text are preserved."""
        ctxs = [
            RetrievedContext(
                text="Line 1\nLine 2\nLine 3", source="test.txt", relevance=0.9
            ),
        ]
        result = _build_context_string(ctxs)
        assert "Line 1\nLine 2\nLine 3" in result

    def test_formats_relevance_with_two_decimals(self) -> None:
        """Test that relevance is formatted to 2 decimal places."""
        ctxs = [
            RetrievedContext(text="Text", source="file.txt", relevance=0.123456),
        ]
        result = _build_context_string(ctxs)
        assert "0.12" in result
        assert "0.123456" not in result

    def test_separator_between_contexts(self) -> None:
        """Test that contexts are separated by horizontal rule."""
        ctxs = [
            RetrievedContext(text="First", source="a.txt", relevance=0.9),
            RetrievedContext(text="Second", source="b.txt", relevance=0.8),
            RetrievedContext(text="Third", source="c.txt", relevance=0.7),
        ]
        result = _build_context_string(ctxs)
        # Should have n-1 separators for n contexts
        assert result.count("---") == 2


class TestModels:
    def test_rag_response_is_frozen(self) -> None:
        r = RAGResponse(answer="test", contexts=[])
        with pytest.raises(AttributeError):
            r.answer = "changed"

    def test_retrieved_context_is_frozen(self) -> None:
        ctx = RetrievedContext(text="t", source="s", relevance=0.5)
        with pytest.raises(AttributeError):
            ctx.text = "changed"
