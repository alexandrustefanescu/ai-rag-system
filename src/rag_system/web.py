"""FastAPI web interface for the RAG system."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from rag_system import rag_engine
from rag_system import vector_store as vs
from rag_system.config import AppConfig
from rag_system.document_loader import load_documents
from rag_system.text_chunker import chunk_documents

logger = logging.getLogger(__name__)

_config = AppConfig()
_client = None
_collection = None

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize ChromaDB client and collection on startup."""
    global _client, _collection
    _client = vs.get_client(_config.vector_store)
    _collection = vs.get_or_create_collection(_client, _config.vector_store)
    logger.info("ChromaDB initialized (%d chunks indexed)", _collection.count())
    yield


app = FastAPI(title="RAG System", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class AskRequest(BaseModel):
    question: str
    model: str | None = None


class SourceResponse(BaseModel):
    text: str
    source: str
    relevance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


class IngestResponse(BaseModel):
    status: str
    chunks: int


class UploadResponse(BaseModel):
    status: str
    files_saved: int
    chunks: int


class DocumentInfo(BaseModel):
    filename: str
    size_kb: float


class DocumentListResponse(BaseModel):
    files: list[DocumentInfo]


class DeleteResponse(BaseModel):
    status: str
    chunks: int


class StatusResponse(BaseModel):
    documents: int
    model: str
    available_models: list[str]
    ollama_connected: bool


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/ingest", response_model=IngestResponse)
async def api_ingest():
    global _collection
    folder = _config.documents_dir
    documents = load_documents(folder)

    if not documents:
        return IngestResponse(status="no_documents", chunks=0)

    chunks = chunk_documents(documents, _config.chunk)
    _collection = vs.reset_collection(_client, _config.vector_store)
    added = vs.add_chunks(_collection, chunks, _config.vector_store.batch_size)

    return IngestResponse(status="ok", chunks=added)


ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}


@app.post("/api/upload", response_model=UploadResponse)
async def api_upload(files: list[UploadFile]):
    global _collection
    docs_dir = Path(_config.documents_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for file in files:
        if not file.filename:
            continue
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue
        dest = docs_dir / file.filename
        content = await file.read()
        dest.write_bytes(content)
        saved += 1

    if saved == 0:
        return UploadResponse(status="no_valid_files", files_saved=0, chunks=0)

    documents = load_documents(str(docs_dir))
    chunks = chunk_documents(documents, _config.chunk)
    _collection = vs.reset_collection(_client, _config.vector_store)
    added = vs.add_chunks(_collection, chunks, _config.vector_store.batch_size)

    return UploadResponse(status="ok", files_saved=saved, chunks=added)


@app.get("/api/documents", response_model=DocumentListResponse)
async def api_list_documents():
    docs_dir = Path(_config.documents_dir)
    if not docs_dir.exists():
        return DocumentListResponse(files=[])

    files = []
    for p in sorted(docs_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append(
                DocumentInfo(
                    filename=p.name,
                    size_kb=round(p.stat().st_size / 1024, 1),
                )
            )
    return DocumentListResponse(files=files)


@app.delete("/api/documents/{filename}", response_model=DeleteResponse)
async def api_delete_document(filename: str):
    global _collection
    docs_dir = Path(_config.documents_dir)
    target = docs_dir / filename

    if not target.is_file() or target.suffix.lower() not in ALLOWED_EXTENSIONS:
        return DeleteResponse(status="not_found", chunks=0)

    target.unlink()

    remaining = load_documents(str(docs_dir))
    _collection = vs.reset_collection(_client, _config.vector_store)
    if remaining:
        chunks = chunk_documents(remaining, _config.chunk)
        added = vs.add_chunks(_collection, chunks, _config.vector_store.batch_size)
    else:
        added = 0

    return DeleteResponse(status="ok", chunks=added)


@app.post("/api/ask", response_model=AskResponse)
async def api_ask(body: AskRequest):
    llm_config = _config.llm
    if body.model and body.model in _config.llm.available_models:
        llm_config = _config.llm.model_copy(update={"model": body.model})

    response = rag_engine.ask(
        body.question,
        _collection,
        config=llm_config,
        n_results=_config.vector_store.query_results,
    )

    sources = [
        SourceResponse(text=ctx.text, source=ctx.source, relevance=ctx.relevance)
        for ctx in response.contexts
    ]

    return AskResponse(answer=response.answer, sources=sources)


@app.get("/api/status", response_model=StatusResponse)
async def api_status():
    connected = True
    try:
        import ollama

        ollama.list()
    except Exception:
        connected = False

    return StatusResponse(
        documents=_collection.count() if _collection else 0,
        model=_config.llm.model,
        available_models=_config.llm.available_models,
        ollama_connected=connected,
    )
