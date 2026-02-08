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
        assert call_kwargs.kwargs["model"] == "tinyllama"


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


class TestModels:
    def test_rag_response_is_frozen(self) -> None:
        r = RAGResponse(answer="test", contexts=[])
        with pytest.raises(AttributeError):
            r.answer = "changed"

    def test_retrieved_context_is_frozen(self) -> None:
        ctx = RetrievedContext(text="t", source="s", relevance=0.5)
        with pytest.raises(AttributeError):
            ctx.text = "changed"
