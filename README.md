# RAG System

A local Retrieval-Augmented Generation system powered by [Ollama](https://ollama.com) and [ChromaDB](https://www.trychroma.com/). Ingest your documents (PDF, TXT, Markdown), then ask questions -- answers are grounded in your own data with source citations.

## Architecture

```
documents/  -->  Document Loader  -->  Text Chunker  -->  ChromaDB (vector store)
                 (.txt .pdf .md)       (overlap chunks)       |
                                                              |  cosine similarity
                                                              v
                 User Query  --------------------------->  Retrieve top-K chunks
                                                              |
                                                              v
                                                     Ollama LLM  -->  Answer
```

## Project Structure

```
ai-rag-system/
├── src/rag_system/          # Source package
│   ├── __init__.py
│   ├── __main__.py          # python -m rag_system entry point
│   ├── cli.py               # CLI (ingest / chat commands)
│   ├── config.py            # Centralized configuration dataclasses
│   ├── models.py            # Domain models (Document, Chunk, RAGResponse)
│   ├── document_loader.py   # PDF, TXT, Markdown loaders
│   ├── text_chunker.py      # Smart boundary-aware text chunking
│   ├── vector_store.py      # ChromaDB operations with embedding caching
│   └── rag_engine.py        # Retrieve + generate pipeline
├── tests/                   # Comprehensive test suite (60 tests)
├── documents/               # Place your documents here
├── scripts/setup.sh         # One-command setup script
├── docker-compose.yml       # Ollama + RAG app containers
├── Dockerfile
├── Makefile                 # Automation commands
└── pyproject.toml           # Project config (uv / pip compatible)
```

## Prerequisites

- **Python 3.11+** (3.12 recommended)
- **[uv](https://docs.astral.sh/uv/)** -- fast Python package manager
- **[Ollama](https://ollama.com)** -- local LLM runtime

## Quick Start

### 1. Setup

```bash
cd ai-rag-system

# Option A: One-command setup
bash scripts/setup.sh

# Option B: Manual setup
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
ollama pull tinyllama
```

### 2. Add Documents

Place `.txt`, `.pdf`, or `.md` files in the `documents/` directory. Two sample documents are included.

### 3. Ingest

```bash
make ingest
# or: python -m rag_system ingest --folder ./documents
```

### 4. Chat

```bash
make chat
# or: python -m rag_system chat --model tinyllama
```

## Docker Usage

Run everything in containers (no local Python/Ollama needed):

```bash
# Start Ollama + RAG services
make docker-up

# Pull the model inside the Ollama container
docker compose exec ollama ollama pull tinyllama

# Ingest documents
make docker-ingest

# Start chatting
make docker-chat

# Stop services
make docker-down
```

## Makefile Commands

Run `make help` to see all available commands:

| Command            | Description                          |
|--------------------|--------------------------------------|
| `make install`     | Install dependencies with uv         |
| `make dev`         | Install with dev dependencies        |
| `make lint`        | Run ruff linter                      |
| `make test`        | Run tests                            |
| `make test-cov`    | Run tests with coverage report       |
| `make ingest`      | Ingest documents from ./documents    |
| `make chat`        | Start interactive chat               |
| `make pull-model`  | Pull the default Ollama model        |
| `make docker-up`   | Start Docker services                |
| `make docker-down` | Stop Docker services                 |
| `make clean`       | Remove caches and build artifacts    |

## Configuration

All settings are centralized in `src/rag_system/config.py` using frozen dataclasses:

| Setting               | Default            | Description                     |
|-----------------------|--------------------|---------------------------------|
| `chunk.size`          | 500                | Characters per chunk            |
| `chunk.overlap`       | 50                 | Overlap between adjacent chunks |
| `vector_store.db_path`| `./chroma_db`      | ChromaDB storage path           |
| `vector_store.embedding_model` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `vector_store.query_results` | 3           | Top-K results per query         |
| `llm.model`           | `tinyllama`        | Ollama model name               |
| `llm.temperature`     | 0.3                | Generation temperature          |
| `llm.max_tokens`      | 512                | Max tokens in response          |

## Testing

```bash
# Run all 60 tests
make test

# Run with coverage
make test-cov

# Run a specific test file
python -m pytest tests/test_text_chunker.py -v
```

## Performance Notes

- **Embedding function caching**: The SentenceTransformer model is loaded once and reused across queries.
- **Batch ingestion**: Documents are added to ChromaDB in batches of 100 to stay within API limits.
- **Smart chunking**: Text is split at paragraph and sentence boundaries to preserve semantic coherence.
- **Frozen dataclasses**: All domain models and config objects are immutable for safety and hashability.
- **OnPush-style design**: The RAG engine returns structured `RAGResponse` objects with parsed contexts, avoiding repeated parsing.

## Supported File Formats

| Format   | Extension | Loader           |
|----------|-----------|------------------|
| Plain text | `.txt`  | Direct read      |
| PDF      | `.pdf`    | pypdf            |
| Markdown | `.md`     | markdown + strip |

## Using a Different Model

```bash
# Pull any Ollama model
ollama pull llama3

# Use it for chat
python -m rag_system chat --model llama3
```

## License

MIT
