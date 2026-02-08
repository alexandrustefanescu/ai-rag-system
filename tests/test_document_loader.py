"""Tests for the document_loader module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from rag_system.document_loader import (
    _load_markdown,
    _load_pdf,
    _load_txt,
    load_documents,
)


class TestLoadTxt:
    def test_reads_file_content(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("Hello, world!", encoding="utf-8")
        assert _load_txt(f) == "Hello, world!"

    def test_handles_unicode(self, tmp_path: Path) -> None:
        f = tmp_path / "unicode.txt"
        f.write_text("Ünïcödé tëxt: 日本語", encoding="utf-8")
        result = _load_txt(f)
        assert "Ünïcödé" in result
        assert "日本語" in result

    def test_handles_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        assert _load_txt(f) == ""


class TestLoadMarkdown:
    def test_strips_html_tags(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("# Heading\n\nSome **bold** text.", encoding="utf-8")
        result = _load_markdown(f)
        assert "Heading" in result
        assert "bold" in result
        assert "<" not in result
        assert ">" not in result

    def test_handles_links(self, tmp_path: Path) -> None:
        f = tmp_path / "links.md"
        f.write_text("[Click here](http://example.com)", encoding="utf-8")
        result = _load_markdown(f)
        assert "Click here" in result


class TestLoadDocuments:
    def test_loads_txt_and_md(self, tmp_docs_dir: Path) -> None:
        docs = load_documents(tmp_docs_dir)
        assert len(docs) == 2
        sources = {d.metadata["source"] for d in docs}
        assert "sample.txt" in sources
        assert "notes.md" in sources

    def test_skips_unsupported_formats(self, tmp_docs_dir: Path) -> None:
        docs = load_documents(tmp_docs_dir)
        sources = {d.metadata["source"] for d in docs}
        assert "image.png" not in sources

    def test_returns_empty_for_empty_folder(self, empty_docs_dir: Path) -> None:
        docs = load_documents(empty_docs_dir)
        assert docs == []

    def test_raises_on_missing_folder(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_documents("/nonexistent/path")

    def test_raises_on_file_instead_of_dir(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("content")
        with pytest.raises(NotADirectoryError):
            load_documents(f)

    def test_metadata_includes_file_type(self, tmp_docs_dir: Path) -> None:
        docs = load_documents(tmp_docs_dir)
        for doc in docs:
            assert "file_type" in doc.metadata
            assert doc.metadata["file_type"] in (".txt", ".md")

    def test_skips_empty_files(self, tmp_path: Path) -> None:
        (tmp_path / "empty.txt").write_text("", encoding="utf-8")
        (tmp_path / "whitespace.txt").write_text("   \n\n  ", encoding="utf-8")
        docs = load_documents(tmp_path)
        assert len(docs) == 0

    def test_content_is_nonempty(self, tmp_docs_dir: Path) -> None:
        docs = load_documents(tmp_docs_dir)
        for doc in docs:
            assert doc.content.strip()

    def test_skips_subdirectories(self, tmp_path: Path) -> None:
        (tmp_path / "subdir").mkdir()
        (tmp_path / "real.txt").write_text("content", encoding="utf-8")
        docs = load_documents(tmp_path)
        assert len(docs) == 1

    def test_loader_exception_skips_file(self, tmp_path: Path) -> None:
        (tmp_path / "good.txt").write_text("valid content", encoding="utf-8")
        (tmp_path / "bad.txt").write_text("will fail", encoding="utf-8")

        with patch("rag_system.document_loader.LOADERS", {
            ".txt": lambda p: (_ for _ in ()).throw(RuntimeError("boom"))
            if p.name == "bad.txt"
            else p.read_text(encoding="utf-8"),
        }):
            docs = load_documents(tmp_path)

        assert len(docs) == 1
        assert docs[0].metadata["source"] == "good.txt"


class TestLoadPdf:
    def test_extracts_text_from_pdf(self, tmp_path: Path) -> None:
        mock_page = type("Page", (), {"extract_text": lambda self: "Hello PDF"})()
        mock_reader = type("Reader", (), {"pages": [mock_page]})()

        with patch("rag_system.document_loader.PdfReader", return_value=mock_reader):
            result = _load_pdf(tmp_path / "test.pdf")

        assert result == "Hello PDF"

    def test_handles_pages_with_no_text(self, tmp_path: Path) -> None:
        page_with_text = type("Page", (), {"extract_text": lambda self: "page one"})()
        page_none = type("Page", (), {"extract_text": lambda self: None})()
        mock_reader = type("Reader", (), {"pages": [page_with_text, page_none]})()

        with patch("rag_system.document_loader.PdfReader", return_value=mock_reader):
            result = _load_pdf(tmp_path / "test.pdf")

        assert "page one" in result
        assert "\n" in result
