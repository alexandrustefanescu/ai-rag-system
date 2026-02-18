"""FastAPI web interface for the RAG system."""

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
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
    application.state.pull_status: dict[str, dict] = {}
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


class GenerationMetricsResponse(BaseModel):
    duration_s: float
    tokens_generated: int
    tokens_per_second: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    metrics: GenerationMetricsResponse | None = None


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
    downloaded_models: list[str]
    ollama_connected: bool


class ModelInfo(BaseModel):
    name: str
    size_mb: float
    downloaded: bool


class ModelListResponse(BaseModel):
    models: list[ModelInfo]


class PullRequest(BaseModel):
    model: str


class PullResponse(BaseModel):
    status: str


class PullStatusResponse(BaseModel):
    status: str
    progress: str


class DeleteModelResponse(BaseModel):
    status: str


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

    # Check that the selected model is actually downloaded.
    selected_model = llm_config.model
    _, downloaded = _get_downloaded_models()
    if not any(
        selected_model == d or d.startswith(selected_model + ":") for d in downloaded
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{selected_model}' is not downloaded. "
            "Pull it from the Models panel first.",
        )

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

    metrics = None
    if response.metrics:
        metrics = GenerationMetricsResponse(
            duration_s=response.metrics.duration_s,
            tokens_generated=response.metrics.tokens_generated,
            tokens_per_second=response.metrics.tokens_per_second,
        )

    return AskResponse(answer=response.answer, sources=sources, metrics=metrics)


@router.post("/ask/stream")
def api_ask_stream(
    body: AskRequest,
    collection=Depends(get_collection),
):
    if collection is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized. Ingest documents first.",
        )

    llm_config = _config.llm
    if body.model and body.model in _config.llm.available_models:
        llm_config = _config.llm.model_copy(
            update={"model": body.model}
        )

    selected_model = llm_config.model
    _, downloaded = _get_downloaded_models()
    if not any(
        selected_model == d or d.startswith(selected_model + ":")
        for d in downloaded
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{selected_model}' is not downloaded. "
            "Pull it from the Models panel first.",
        )

    return StreamingResponse(
        rag_engine.stream_answer(
            body.question,
            collection,
            config=llm_config,
            n_results=_config.vector_store.query_results,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _get_downloaded_models() -> tuple[bool, list[str]]:
    """Return (ollama_connected, list_of_model_names)."""
    try:
        import ollama

        response = ollama.list()
        models = [m.model for m in (response.models or [])]
        return True, models
    except Exception:
        return False, []


@router.get("/status", response_model=StatusResponse)
def api_status(collection=Depends(get_collection)):
    connected, downloaded = _get_downloaded_models()

    return StatusResponse(
        documents=collection.count() if collection else 0,
        model=_config.llm.model,
        available_models=_config.llm.available_models,
        downloaded_models=downloaded,
        ollama_connected=connected,
    )


@router.get("/models", response_model=ModelListResponse)
def api_list_models():
    connected, downloaded = _get_downloaded_models()

    # Build a size lookup from downloaded models.
    size_map: dict[str, float] = {}
    if connected:
        try:
            import ollama

            response = ollama.list()
            for m in response.models or []:
                size_mb = round((m.size or 0) / 1024 / 1024, 1)
                size_map[m.model] = size_mb
        except Exception:
            pass

    models = []
    for name in _config.llm.available_models:
        is_downloaded = any(name == d or d.startswith(name + ":") for d in downloaded)
        models.append(
            ModelInfo(
                name=name,
                size_mb=size_map.get(name, 0),
                downloaded=is_downloaded,
            )
        )

    return ModelListResponse(models=models)


def _pull_model_background(app_state: object, model: str) -> None:
    """Pull a model in a background thread, updating app.state.pull_status."""
    try:
        import ollama

        pull_status = getattr(app_state, "pull_status", {})
        pull_status[model] = {"status": "pulling", "progress": "starting..."}

        for progress in ollama.pull(model, stream=True):
            status_str = getattr(progress, "status", "") or ""
            total = getattr(progress, "total", 0) or 0
            completed = getattr(progress, "completed", 0) or 0
            if total > 0:
                pct = round(completed / total * 100)
                pull_status[model] = {
                    "status": "pulling",
                    "progress": f"{status_str} {pct}%",
                }
            else:
                pull_status[model] = {
                    "status": "pulling",
                    "progress": status_str,
                }

        pull_status[model] = {"status": "completed", "progress": "done"}
    except Exception as exc:
        pull_status = getattr(app_state, "pull_status", {})
        pull_status[model] = {"status": "error", "progress": str(exc)}


@router.post("/models/pull", response_model=PullResponse)
def api_pull_model(body: PullRequest, request: Request):
    if body.model not in _config.llm.available_models:
        raise HTTPException(status_code=400, detail="Model not in available list.")

    pull_status = getattr(request.app.state, "pull_status", {})
    current = pull_status.get(body.model, {})
    if current.get("status") == "pulling":
        return PullResponse(status="already_pulling")

    thread = threading.Thread(
        target=_pull_model_background,
        args=(request.app.state, body.model),
        daemon=True,
    )
    thread.start()

    return PullResponse(status="pulling")


@router.get("/models/{model_name:path}/status", response_model=PullStatusResponse)
def api_model_status(model_name: str, request: Request):
    pull_status = getattr(request.app.state, "pull_status", {})
    info = pull_status.get(model_name)

    if info is None:
        # Check if already downloaded.
        _, downloaded = _get_downloaded_models()
        is_downloaded = any(
            model_name == d or d.startswith(model_name + ":") for d in downloaded
        )
        if is_downloaded:
            return PullStatusResponse(status="completed", progress="done")
        return PullStatusResponse(status="not_started", progress="")

    return PullStatusResponse(status=info["status"], progress=info["progress"])


@router.delete(
    "/models/{model_name:path}",
    response_model=DeleteModelResponse,
)
def api_delete_model(model_name: str, request: Request):
    """Remove a downloaded model from Ollama."""
    _, downloaded = _get_downloaded_models()
    is_downloaded = any(
        model_name == d or d.startswith(model_name + ":") for d in downloaded
    )
    if not is_downloaded:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' is not downloaded.",
        )

    try:
        import ollama

        ollama.delete(model_name)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete model: {exc}",
        )

    # Clear any stale pull status for this model.
    pull_status = getattr(request.app.state, "pull_status", {})
    pull_status.pop(model_name, None)

    return DeleteModelResponse(status="ok")


app.include_router(router)

# Serve index.html as a static file (must be last — catches all unmatched paths)
app.mount("/", StaticFiles(directory=str(TEMPLATES_DIR), html=True), name="ui")
