"""RAG engine — retrieves context from the vector store and
generates answers via Ollama."""

import logging

import ollama

from rag_system import vector_store as vs
from rag_system.config import LLMConfig
from rag_system.models import RAGResponse, RetrievedContext

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided context. "
    "Use only the information from the context to answer. "
    "The context may contain passages from multiple source documents — "
    "focus on the passages most relevant to the question and ignore unrelated ones. "
    "If the context doesn't contain enough information, say so clearly. "
    "Always cite the source file for each piece of information you use."
)


def _parse_results(results: dict) -> list[RetrievedContext]:
    """Convert raw ChromaDB results into RetrievedContext objects.

    Cosine distances are converted to similarity scores (1 - distance).

    Args:
        results: Raw dict returned by ``chromadb.Collection.query``,
            containing ``documents``, ``metadatas``, and ``distances``.

    Returns:
        List of RetrievedContext objects with text, source, and relevance.
    """
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    contexts: list[RetrievedContext] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        contexts.append(
            RetrievedContext(
                text=doc,
                source=meta.get("source", "unknown"),
                relevance=round(1 - dist, 4),  # cosine distance → similarity
            )
        )
    return contexts


def _build_context_string(contexts: list[RetrievedContext]) -> str:
    """Format retrieved contexts into a prompt-ready string.

    Each context is labelled with its source file and relevance score,
    separated by horizontal rules.

    Args:
        contexts: Retrieved context passages to format.

    Returns:
        A single string ready to be injected into the LLM prompt.
    """
    parts = [
        f"[Source: {ctx.source} | Relevance: {ctx.relevance:.2f}]\n{ctx.text}"
        for ctx in contexts
    ]
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, context: str, config: LLMConfig | None = None) -> str:
    """Generate an answer using the Ollama LLM.

    Sends a system prompt and a user prompt (context + question) to
    the configured Ollama model and returns the generated text.

    Args:
        query: The user's question.
        context: Pre-formatted context string from retrieved documents.
        config: LLM settings (model, temperature, max_tokens).
            Uses defaults if not provided.

    Returns:
        The model's generated answer as a string.
    """
    cfg = config or LLMConfig()
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    response = ollama.chat(
        model=cfg.model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": cfg.temperature, "num_predict": cfg.max_tokens},
    )
    return response["message"]["content"]


_RELEVANCE_THRESHOLD = 0.3


def ask(
    query: str,
    collection,
    config: LLMConfig | None = None,
    n_results: int = 5,
) -> RAGResponse:
    """Full RAG pipeline: retrieve, filter, build context, generate answer.

    Queries the vector store for relevant chunks, filters out low-relevance
    results, builds a context string, and sends it to the LLM. If no
    relevant documents are found, returns a fallback message without
    calling the LLM.

    Args:
        query: The user's natural-language question.
        collection: ChromaDB collection to search.
        config: LLM settings. Uses defaults if not provided.
        n_results: Number of context chunks to retrieve before filtering.

    Returns:
        A RAGResponse containing the answer and the retrieved contexts.
    """
    results = vs.query(collection, query, n_results=n_results)
    contexts = _parse_results(results)

    # Filter out low-relevance chunks to reduce noise.
    contexts = [c for c in contexts if c.relevance >= _RELEVANCE_THRESHOLD]

    if not contexts:
        return RAGResponse(
            answer="No relevant documents found. Try ingesting documents first.",
            contexts=[],
        )

    context_str = _build_context_string(contexts)
    answer = generate_answer(query, context_str, config)

    return RAGResponse(answer=answer, contexts=contexts)
