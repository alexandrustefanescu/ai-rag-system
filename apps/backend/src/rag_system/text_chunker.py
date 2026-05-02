"""Text chunker — splits documents into overlapping chunks with smart boundaries."""

import re

import numpy as np

from rag_system.config import ChunkConfig
from rag_system.models import Chunk, Document

# Sentence-ending delimiters, ordered by preference.
_SENTENCE_BREAKS = (". ", "! ", "? ", "\n")

# Sentence boundary regex — handles common cases without spaCy/NLTK.
_SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])"
    r"|(?<=[.!?][\"'])\s+(?=[A-Z])"
    r"|\n\n+",
)


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


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex patterns.

    Falls back to 500-char character splits for sentences longer than 1000
    characters (e.g. text with no punctuation).

    Args:
        text: Input text to split.

    Returns:
        List of non-empty sentence strings.
    """
    if not text.strip():
        return []

    parts = _SENTENCE_RE.split(text)
    result: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > 1000:
            for i in range(0, len(part), 500):
                segment = part[i : i + 500].strip()
                if segment:
                    result.append(segment)
        else:
            result.append(part)
    return result


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def _chunk_text_semantic(
    text: str,
    threshold: float = 0.5,
    max_size: int = 2000,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
) -> list[str]:
    """Split text into chunks based on semantic similarity between sentences.

    Groups consecutive sentences together until a topic shift is detected
    (cosine similarity between consecutive sentences drops below *threshold*)
    or the accumulated chunk exceeds *max_size* characters.

    Args:
        text: Source text to chunk.
        threshold: Minimum cosine similarity to keep sentences in the same
            chunk.  Lower values produce fewer, larger chunks.
        max_size: Hard upper limit on chunk size in characters.
        embedding_model: HuggingFace model name (reuses the cached instance
            from :func:`rag_system.vector_store.get_embedding_function`).

    Returns:
        Ordered list of semantic chunks.
    """
    if not text.strip():
        return []

    if len(text) <= max_size:
        return [text.strip()]

    sentences = _split_sentences(text)

    if len(sentences) <= 1:
        return chunk_text(text, chunk_size=max_size, chunk_overlap=100)

    from rag_system.vector_store import get_embedding_function

    embed_fn = get_embedding_function(embedding_model)
    embeddings = embed_fn(sentences)

    chunks: list[str] = []
    current_sentences: list[str] = [sentences[0]]
    current_size = len(sentences[0])

    for i in range(1, len(sentences)):
        similarity = _cosine_similarity(
            np.asarray(embeddings[i - 1]),
            np.asarray(embeddings[i]),
        )
        sent_len = len(sentences[i])

        # Account for space separators when joining sentences.
        joined_size = current_size + len(current_sentences) - 1 + 1 + sent_len

        if similarity < threshold or joined_size > max_size:
            chunk = " ".join(current_sentences).strip()
            if chunk:
                chunks.append(chunk)
            current_sentences = [sentences[i]]
            current_size = sent_len
        else:
            current_sentences.append(sentences[i])
            current_size += sent_len

    if current_sentences:
        chunk = " ".join(current_sentences).strip()
        if chunk:
            chunks.append(chunk)

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
        if config.strategy == "semantic":
            texts = _chunk_text_semantic(
                doc.content,
                threshold=config.semantic_threshold,
                max_size=config.semantic_max_size,
            )
        else:
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
