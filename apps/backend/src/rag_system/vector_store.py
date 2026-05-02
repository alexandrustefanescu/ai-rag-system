"""Vector store — manages ChromaDB collections and semantic search."""

import logging

import chromadb
from chromadb.utils import embedding_functions

from rag_system.config import VectorStoreConfig
from rag_system.models import Chunk

logger = logging.getLogger(__name__)

# Module-level cache to avoid re-creating the embedding function repeatedly.
_embedding_fn_cache: dict[str, object] = {}


def get_client(config: VectorStoreConfig | None = None) -> chromadb.PersistentClient:
    """Return a ChromaDB persistent client.

    Args:
        config: Vector store settings (db path, etc.).
            Uses defaults if not provided.

    Returns:
        A ChromaDB PersistentClient connected to the configured path.
    """
    cfg = config or VectorStoreConfig()
    return chromadb.PersistentClient(path=cfg.db_path)


def get_embedding_function(
    model_name: str = "all-MiniLM-L6-v2",
) -> embedding_functions.SentenceTransformerEmbeddingFunction:
    """Return a cached SentenceTransformer embedding function.

    Embedding functions are cached at module level so the model is
    loaded only once per model name, regardless of how many times
    this function is called.

    Args:
        model_name: HuggingFace model identifier for the embedding model.

    Returns:
        A SentenceTransformerEmbeddingFunction instance (cached).
    """
    if model_name not in _embedding_fn_cache:
        _embedding_fn_cache[model_name] = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name,
            )
        )
    return _embedding_fn_cache[model_name]  # type: ignore[return-value]


def get_or_create_collection(
    client: chromadb.PersistentClient,
    config: VectorStoreConfig | None = None,
) -> chromadb.Collection:
    """Get or create a ChromaDB collection with cosine similarity.

    Args:
        client: An active ChromaDB persistent client.
        config: Vector store settings (collection name, embedding model).
            Uses defaults if not provided.

    Returns:
        A ChromaDB Collection configured with cosine distance.
    """
    cfg = config or VectorStoreConfig()
    ef = get_embedding_function(cfg.embedding_model)
    return client.get_or_create_collection(
        name=cfg.collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(
    collection: chromadb.Collection,
    chunks: list[Chunk],
    batch_size: int = 100,
) -> int:
    """Add chunks to the collection in batches.

    Chunk IDs are generated sequentially starting from the current
    collection size, so new chunks never overwrite existing ones.

    Args:
        collection: The target ChromaDB collection.
        chunks: Text chunks to add (with metadata).
        batch_size: Maximum number of chunks per ChromaDB upsert call.

    Returns:
        Number of chunks successfully added (0 if the list is empty).
    """
    if not chunks:
        return 0

    offset = collection.count()
    ids = [f"chunk_{offset + i}" for i in range(len(chunks))]
    texts = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]

    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.add(
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )

    logger.info("Added %d chunks to the vector store.", len(chunks))
    return len(chunks)


def query(
    collection: chromadb.Collection,
    query_text: str,
    n_results: int = 3,
) -> dict:
    """Query the collection for the most relevant chunks.

    Args:
        collection: The ChromaDB collection to search.
        query_text: The natural-language query string.
        n_results: Maximum number of results to return.

    Returns:
        Raw ChromaDB query result dict containing ``documents``,
        ``metadatas``, and ``distances`` keys.
    """
    return collection.query(query_texts=[query_text], n_results=n_results)


def reset_collection(
    client: chromadb.PersistentClient,
    config: VectorStoreConfig | None = None,
) -> chromadb.Collection:
    """Delete and recreate the collection (useful for re-ingestion).

    If the collection does not exist yet, the delete is silently ignored.

    Args:
        client: An active ChromaDB persistent client.
        config: Vector store settings. Uses defaults if not provided.

    Returns:
        A fresh, empty ChromaDB Collection.
    """
    cfg = config or VectorStoreConfig()
    try:
        client.delete_collection(cfg.collection_name)
    except Exception:
        pass
    return get_or_create_collection(client, cfg)
