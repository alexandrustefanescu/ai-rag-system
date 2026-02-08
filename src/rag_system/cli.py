"""CLI interface for the RAG system."""

import argparse
import logging
import sys

from rag_system.config import AppConfig, ChunkConfig, LLMConfig
from rag_system.document_loader import load_documents
from rag_system.text_chunker import chunk_documents
from rag_system import vector_store as vs
from rag_system import rag_engine


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def ingest(folder_path: str, config: AppConfig | None = None) -> None:
    """Ingest documents from a folder into the vector store.

    Loads all supported files (.txt, .pdf, .md) from the given folder,
    splits them into chunks, and stores the embeddings in ChromaDB.
    The existing collection is reset before ingestion.

    Args:
        folder_path: Path to the directory containing documents.
        config: Application configuration. Uses defaults if not provided.
    """
    cfg = config or AppConfig()

    print(f"\n📂 Loading documents from: {folder_path}")
    documents = load_documents(folder_path)

    if not documents:
        print("No supported documents found (.txt, .pdf, .md)")
        return

    print(f"\n✂️  Chunking {len(documents)} document(s)...")
    chunks = chunk_documents(documents, cfg.chunk)
    print(f"  Created {len(chunks)} chunks")

    print("\n💾 Storing in ChromaDB...")
    client = vs.get_client(cfg.vector_store)
    collection = vs.reset_collection(client, cfg.vector_store)
    added = vs.add_chunks(collection, chunks, cfg.vector_store.batch_size)

    print(f"\n✅ Ingestion complete! ({added} chunks stored)")


def chat(config: AppConfig | None = None) -> None:
    """Start an interactive chat session.

    Connects to the ChromaDB vector store and enters a REPL loop where
    the user can ask questions. Each query is answered using the RAG
    pipeline (retrieve context + generate via Ollama). Exits on 'quit',
    'exit', 'q', EOF, or KeyboardInterrupt.

    Args:
        config: Application configuration. Uses defaults if not provided.
    """
    cfg = config or AppConfig()

    client = vs.get_client(cfg.vector_store)
    collection = vs.get_or_create_collection(client, cfg.vector_store)

    if collection.count() == 0:
        print("No documents in the vector store. Run ingestion first:")
        print("  python -m rag_system ingest --folder ./documents")
        return

    print(f"\n📚 RAG Chat ({collection.count()} chunks indexed)")
    print(f"🤖 Using Ollama model: {cfg.llm.model}")
    print("\nType your question (or 'quit' to exit):\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        response = rag_engine.ask(
            query, collection, config=cfg.llm, n_results=cfg.vector_store.query_results
        )
        print(f"\nAssistant:\n{response.answer}\n")


def main() -> None:
    """CLI entry point — parse arguments and dispatch to ingest or chat."""
    parser = argparse.ArgumentParser(
        description="RAG System — Local LLM with Ollama",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ingest
    ingest_p = subparsers.add_parser("ingest", help="Ingest documents from a folder")
    ingest_p.add_argument(
        "--folder", type=str, default="./documents", help="Documents folder path"
    )

    # chat
    chat_p = subparsers.add_parser("chat", help="Start interactive chat")
    chat_p.add_argument(
        "--model", type=str, default="tinyllama", help="Ollama model name"
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command == "ingest":
        ingest(args.folder)
    elif args.command == "chat":
        cfg = AppConfig(llm=LLMConfig(model=args.model))
        chat(cfg)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
