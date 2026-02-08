.PHONY: help install dev lint test test-cov ingest chat docker-up docker-down docker-ingest docker-chat clean pull-model

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Local Development ──────────────────────────────────────────────

install: ## Install dependencies with uv
	uv pip install -e .

dev: ## Install with dev dependencies
	uv pip install -e ".[dev]"

lint: ## Run ruff linter
	uv run ruff check src/ tests/

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage report
	uv run pytest --cov=rag_system --cov-report=term-missing

# ── RAG Operations ─────────────────────────────────────────────────

pull-model: ## Pull the default Ollama model (tinyllama)
	ollama pull tinyllama

ingest: ## Ingest documents from ./documents
	uv run python -m rag_system ingest --folder ./documents

chat: ## Start interactive chat
	uv run python -m rag_system chat

# ── Docker ─────────────────────────────────────────────────────────

docker-up: ## Start all services (Ollama + RAG app)
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

docker-ingest: ## Run ingestion inside Docker
	docker compose run --rm rag ingest --folder ./documents

docker-chat: ## Start chat inside Docker
	docker compose run --rm rag chat

# ── Utilities ──────────────────────────────────────────────────────

clean: ## Remove caches and build artifacts
	rm -rf __pycache__ .pytest_cache .ruff_cache chroma_db/ dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
