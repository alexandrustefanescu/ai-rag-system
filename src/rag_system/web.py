"""FastAPI web interface for the RAG system."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
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


class AskRequest(BaseModel):
    question: str


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


class StatusResponse(BaseModel):
    documents: int
    model: str
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


@app.post("/api/ask", response_model=AskResponse)
async def api_ask(body: AskRequest):
    response = rag_engine.ask(
        body.question,
        _collection,
        config=_config.llm,
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
        ollama_connected=connected,
    )
