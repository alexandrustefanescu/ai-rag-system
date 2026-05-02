"""Document loader — reads PDF, TXT, and Markdown files from a directory."""

import logging
import re
from pathlib import Path
from typing import Callable

import markdown
from pypdf import PdfReader

from rag_system.models import Document

logger = logging.getLogger(__name__)


def _load_txt(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def _load_pdf(file_path: Path) -> str:
    """Extract text from a PDF file.

    Reads every page and joins them with newlines. Pages that yield
    no text (scanned images, empty pages) are treated as empty strings.

    Args:
        file_path: Path to the PDF file.

    Returns:
        The concatenated text of all pages.
    """
    reader = PdfReader(str(file_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _load_markdown(file_path: Path) -> str:
    """Load a Markdown file and return its plain-text content.

    Converts the Markdown source to HTML, then strips all HTML tags
    to produce clean text suitable for embedding.

    Args:
        file_path: Path to the Markdown file.

    Returns:
        Plain text with all Markdown/HTML formatting removed.
    """
    raw = file_path.read_text(encoding="utf-8")
    html = markdown.markdown(raw)
    return re.sub(r"<[^>]+>", "", html)


# Supported file extensions mapped to their loader functions.
LOADERS: dict[str, Callable[[Path], str]] = {
    ".txt": _load_txt,
    ".pdf": _load_pdf,
    ".md": _load_markdown,
}


def load_documents(folder_path: str | Path) -> list[Document]:
    """Load all supported documents from a folder.

    Iterates over files in the directory, loading each with the
    appropriate loader based on extension. Empty files and files
    that fail to load are skipped with a warning.

    Args:
        folder_path: Path to the directory containing documents.

    Returns:
        List of Document objects sorted by filename, each containing
        the file's text content and metadata (source, file_type).

    Raises:
        FileNotFoundError: If the folder does not exist.
        NotADirectoryError: If the path is not a directory.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder_path}")

    documents: list[Document] = []

    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        loader = LOADERS.get(ext)
        if loader is None:
            continue

        try:
            content = loader(file_path)
            if not content.strip():
                logger.warning("Skipping empty file: %s", file_path.name)
                continue

            documents.append(
                Document(
                    content=content,
                    metadata={"source": file_path.name, "file_type": ext},
                )
            )
            logger.info("Loaded: %s", file_path.name)
        except Exception:
            logger.exception("Failed to load %s", file_path.name)

    return documents
