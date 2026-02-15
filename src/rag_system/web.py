"""FastAPI web interface for the RAG system."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_system import rag_engine
from rag_system import vector_store as vs
from rag_system.config import AppConfig
from rag_system.document_loader import load_documents
from rag_system.text_chunker import chunk_documents

logger = logging.getLogger(__name__)

_config = AppConfig()

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialize ChromaDB client and collection on startup."""
    client = vs.get_client(_config.vector_store)
    collection = vs.get_or_create_collection(client, _config.vector_store)
    application.state.chroma_client = client
    application.state.chroma_collection = collection
    logger.info("ChromaDB initialized (%d chunks indexed)", collection.count())
    yield


app = FastAPI(
    title="RAG System",
    lifespan=lifespan,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

router = APIRouter(prefix="/api/v1")


def get_collection(request: Request):
    """FastAPI dependency — return the current ChromaDB collection from app state."""
    return getattr(request.app.state, "chroma_collection", None)


def get_client(request: Request):
    """FastAPI dependency — return the current ChromaDB client from app state."""
    return getattr(request.app.state, "chroma_client", None)


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
    chunk_count: int = 0


class DocumentListResponse(BaseModel):
    files: list[DocumentInfo]


class DeleteResponse(BaseModel):
    status: str
    chunks: int


class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    documents: int


class StatusResponse(BaseModel):
    documents: int
    model: str
    available_models: list[str]
    ollama_connected: bool


@router.get("/health", response_model=HealthResponse)
async def api_health(collection=Depends(get_collection)):
    connected = True
    try:
        import ollama

        ollama.list()
    except Exception:
        connected = False

    doc_count = collection.count() if collection else 0
    status = "healthy" if connected else "degraded"

    return HealthResponse(
        status=status,
        ollama_connected=connected,
        documents=doc_count,
    )


@router.post("/ingest", response_model=IngestResponse)
def api_ingest(
    request: Request,
    collection=Depends(get_collection),
    client=Depends(get_client),
):
    folder = _config.documents_dir
    documents = load_documents(folder)

    if not documents:
        return IngestResponse(status="no_documents", chunks=0)

    chunks = chunk_documents(documents, _config.chunk)
    collection = vs.reset_collection(client, _config.vector_store)
    request.app.state.chroma_collection = collection
    added = vs.add_chunks(collection, chunks, _config.vector_store.batch_size)

    return IngestResponse(status="ok", chunks=added)


ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}


@router.post("/upload", response_model=UploadResponse)
def api_upload(
    files: list[UploadFile],
    request: Request,
    collection=Depends(get_collection),
    client=Depends(get_client),
):
    docs_dir = Path(_config.documents_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    docs_dir_resolved = docs_dir.resolve()
    for file in files:
        if not file.filename:
            continue
        # Use only the basename to prevent path traversal.
        safe_name = Path(file.filename).name
        if not safe_name:
            continue
        ext = Path(safe_name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue
        dest = (docs_dir_resolved / safe_name).resolve()
        try:
            dest.relative_to(docs_dir_resolved)
        except ValueError:
            continue
        max_size = 50 * 1024 * 1024  # 50 MB limit per file
        read_chunk = 1024 * 1024  # 1 MB read chunks
        total = 0
        parts: list[bytes] = []
        oversized = False
        while True:
            chunk = file.file.read(read_chunk)
            if not chunk:
                break
            total += len(chunk)
            if total > max_size:
                oversized = True
                break
            parts.append(chunk)
        if oversized:
            continue
        dest.write_bytes(b"".join(parts))
        saved += 1

    if saved == 0:
        return UploadResponse(status="no_valid_files", files_saved=0, chunks=0)

    documents = load_documents(str(docs_dir))
    chunks = chunk_documents(documents, _config.chunk)
    collection = vs.reset_collection(client, _config.vector_store)
    request.app.state.chroma_collection = collection
    added = vs.add_chunks(collection, chunks, _config.vector_store.batch_size)

    return UploadResponse(status="ok", files_saved=saved, chunks=added)


@router.get("/documents", response_model=DocumentListResponse)
def api_list_documents(collection=Depends(get_collection)):
    files = []

    # List documents from the vector store metadata (the source of truth).
    if collection and collection.count() > 0:
        result = collection.get(include=["metadatas"])
        sources: dict[str, int] = {}
        for meta in result.get("metadatas") or []:
            src = (meta or {}).get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1

        docs_dir = Path(_config.documents_dir)
        for source, chunk_count in sorted(sources.items()):
            path = docs_dir / source
            size_kb = round(path.stat().st_size / 1024, 1) if path.is_file() else 0
            files.append(
                DocumentInfo(
                    filename=source,
                    size_kb=size_kb,
                    chunk_count=chunk_count,
                )
            )

    return DocumentListResponse(files=files)


@router.delete("/documents/{filename}", response_model=DeleteResponse)
def api_delete_document(filename: str, collection=Depends(get_collection)):
    docs_dir = Path(_config.documents_dir).resolve()
    target = (docs_dir / filename).resolve()

    # Prevent path traversal — target must be inside documents dir.
    try:
        target.relative_to(docs_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Delete the file from disk if it exists.
    if target.is_file():
        target.unlink()

    # Remove chunks with this source from the vector store.
    if collection and collection.count() > 0:
        result = collection.get(include=["metadatas"])
        ids_to_delete = [
            doc_id
            for doc_id, meta in zip(
                result.get("ids") or [],
                result.get("metadatas") or [],
            )
            if (meta or {}).get("source") == filename
        ]
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)

    remaining_chunks = collection.count() if collection else 0
    return DeleteResponse(status="ok", chunks=remaining_chunks)


@router.post("/ask", response_model=AskResponse)
def api_ask(body: AskRequest, collection=Depends(get_collection)):
    if collection is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized. Ingest documents first.",
        )

    llm_config = _config.llm
    if body.model and body.model in _config.llm.available_models:
        llm_config = _config.llm.model_copy(update={"model": body.model})

    response = rag_engine.ask(
        body.question,
        collection,
        config=llm_config,
        n_results=_config.vector_store.query_results,
    )

    sources = [
        SourceResponse(text=ctx.text, source=ctx.source, relevance=ctx.relevance)
        for ctx in response.contexts
    ]

    return AskResponse(answer=response.answer, sources=sources)


@router.get("/status", response_model=StatusResponse)
async def api_status(collection=Depends(get_collection)):
    connected = True
    try:
        import ollama

        ollama.list()
    except Exception:
        connected = False

    return StatusResponse(
        documents=collection.count() if collection else 0,
        model=_config.llm.model,
        available_models=_config.llm.available_models,
        ollama_connected=connected,
    )


app.include_router(router)

# Serve index.html as a static file (must be last — catches all unmatched paths)
app.mount("/", StaticFiles(directory=str(TEMPLATES_DIR), html=True), name="ui")
