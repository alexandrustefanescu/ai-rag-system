"""Text chunker — splits documents into overlapping chunks with smart boundaries."""

from rag_system.config import ChunkConfig
from rag_system.models import Chunk, Document

# Sentence-ending delimiters, ordered by preference.
_SENTENCE_BREAKS = (". ", "! ", "? ", "\n")


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Split text into chunks of approximately *chunk_size* characters.

    Tries to break at paragraph (``\\n\\n``) or sentence boundaries
    (``. ``, ``! ``, ``? ``, ``\\n``) when possible. Adjacent chunks
    share *chunk_overlap* characters for context continuity.

    Args:
        text: The source text to split.
        chunk_size: Target maximum number of characters per chunk.
        chunk_overlap: Number of characters shared between adjacent chunks.

    Returns:
        Ordered list of text chunks. Returns an empty list if the
        input is empty or whitespace-only.
    """
    if not text.strip():
        return []

    if len(text) <= chunk_size:
        return [text.strip()]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Prefer paragraph break
            newline_pos = text.rfind("\n\n", start, end)
            if newline_pos > start + chunk_size // 2:
                end = newline_pos + 2
            else:
                # Fall back to sentence break
                for sep in _SENTENCE_BREAKS:
                    sep_pos = text.rfind(sep, start, end)
                    if sep_pos > start + chunk_size // 2:
                        end = sep_pos + len(sep)
                        break

        segment = text[start:end].strip()
        if segment:
            chunks.append(segment)

        start = end - chunk_overlap

    return chunks


def chunk_documents(
    documents: list[Document],
    config: ChunkConfig | None = None,
) -> list[Chunk]:
    """Chunk all documents and return a flat list of Chunk objects.

    Each document is split using ``chunk_text``, and the resulting
    chunks inherit the original document's metadata plus ``chunk_index``
    and ``total_chunks`` fields.

    Args:
        documents: Source documents to chunk.
        config: Chunking parameters (size, overlap). Uses defaults
            if not provided.

    Returns:
        Flat list of Chunk objects preserving document order.
    """
    if config is None:
        config = ChunkConfig()

    result: list[Chunk] = []

    for doc in documents:
        texts = chunk_text(doc.content, config.size, config.overlap)
        for idx, text in enumerate(texts):
            result.append(
                Chunk(
                    text=text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": idx,
                        "total_chunks": len(texts),
                    },
                )
            )

    return result
