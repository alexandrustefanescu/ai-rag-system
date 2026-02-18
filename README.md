# AI RAG System

A local Retrieval-Augmented Generation system powered by [Ollama](https://ollama.com) and [ChromaDB](https://www.trychroma.com/). Ingest your documents (PDF, TXT, Markdown), then ask questions -- answers are grounded in your own data with source citations.

## Install

One command. Works on Raspberry Pi, Ubuntu, Debian, Fedora, and macOS. Docker is installed automatically if needed.

```bash
curl -fsSL https://alexandrustefanescu.github.io/ai-rag-system/install.sh | bash
```

Then open **https://localhost:8443** in your browser.

> The app uses a self-signed SSL certificate. Your browser will show a security warning -- click **Advanced** then **Proceed** to continue.

## How It Works

```
Documents  -->  Chunker  -->  ChromaDB (vectors)
                                |
User Query  -->  Retrieve top chunks  -->  Ollama LLM  -->  Answer + Sources
```

1. Upload documents (PDF, TXT, or Markdown)
2. Documents are split into chunks and stored as vectors
3. When you ask a question, the most relevant chunks are retrieved
4. The LLM generates an answer grounded in those chunks
5. Sources are cited so you can verify the answer

## Add Your Documents

**Through the web interface:** Open https://localhost:8443, click the upload area or drag-and-drop your files.

**Via the command line:**

```bash
cp ~/my-notes.pdf ~/ai-rag-system/documents/
curl -k -X POST https://localhost:8443/api/v1/ingest
```

Supported formats: `.txt`, `.pdf`, `.md`

## Web Interface

- **Chat** -- ask questions and get answers with source citations
- **Chat history** -- multiple conversations stored locally, with titles and timestamps
- **Model selector** -- switch between gemma3:1b, llama3.2:1b, etc.
- **Models panel** -- download and delete Ollama models from the UI
- **Upload** -- drag-and-drop files to ingest
- **Document management** -- view indexed files, chunk counts, delete documents

## API

Interactive docs at https://localhost:8443/api/docs

| Method   | Endpoint                           | Description                    |
|----------|------------------------------------|--------------------------------|
| `GET`    | `/api/v1/health`                   | Health check                   |
| `GET`    | `/api/v1/status`                   | System status                  |
| `POST`   | `/api/v1/ask`                      | Ask a question                 |
| `POST`   | `/api/v1/upload`                   | Upload documents               |
| `POST`   | `/api/v1/ingest`                   | Re-ingest documents folder     |
| `GET`    | `/api/v1/documents`                | List indexed documents         |
| `DELETE` | `/api/v1/documents/{filename}`     | Delete a document              |
| `GET`    | `/api/v1/models`                   | List available/downloaded models |
| `POST`   | `/api/v1/models/pull`              | Download a model               |
| `GET`    | `/api/v1/models/{model_name}/status` | Check model pull status      |
| `DELETE` | `/api/v1/models/{model_name}`      | Delete a downloaded model      |

```bash
# Ask a question
curl -k -X POST https://localhost:8443/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'

# Upload a document
curl -k -X POST https://localhost:8443/api/v1/upload \
  -F "files=@my_document.pdf"
```

## Configuration

Edit `~/ai-rag-system/docker-compose.yml` to customize, then restart:

```bash
cd ~/ai-rag-system && docker compose restart rag
```

| Variable               | Default                  | Description              |
|------------------------|--------------------------|--------------------------|
| `LLM_MODEL`           | `gemma3:1b`              | Default LLM model        |
| `LLM_AVAILABLE_MODELS`| `gemma3:1b, llama3.2:1b` | Models in dropdown       |
| `LLM_TEMPERATURE`     | `0.3`                    | Generation temperature   |
| `LLM_MAX_TOKENS`      | `512`                    | Max response tokens      |
| `CHUNK_SIZE`           | `500`                    | Characters per chunk     |
| `CHUNK_OVERLAP`        | `100`                    | Overlap between chunks   |
| `VS_QUERY_RESULTS`    | `5`                      | Candidates per query     |

## Useful Commands

```bash
cd ~/ai-rag-system

docker compose logs -f       # View logs
docker compose restart       # Restart
docker compose down          # Stop
docker compose down -v       # Uninstall (removes all data)
```

## System Requirements

|              | Minimum              | Recommended                      |
|--------------|----------------------|----------------------------------|
| **RAM**      | 4 GB                 | 8 GB                             |
| **Disk**     | 4 GB free            | 8 GB free                        |
| **OS**       | Linux (ARM64/x86) or macOS | Raspberry Pi OS, Ubuntu, Debian |

## License

MIT
