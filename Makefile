.PHONY: help install install-dev dev lint format test test-cov test-e2e test-e2e-headed playwright-install ingest chat pull-model docker-up docker-down docker-ingest docker-chat backup healthcheck healthcheck-watch generate-certs build clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Local Development ──────────────────────────────────────────────

install: ## Install dependencies with uv
	uv pip install -e .

install-dev: ## Install with dev dependencies
	uv pip install -e ".[dev]"

generate-certs: ## Generate self-signed TLS certificates for local HTTPS
	bash scripts/generate-certs.sh

dev: generate-certs ## Run the app in dev mode (HTTPS)
	uv run uvicorn rag_system.web:app --reload --host 0.0.0.0 --port 8443 --ssl-keyfile ./certs/key.pem --ssl-certfile ./certs/cert.pem

lint: ## Run ruff linter and auto-fix issues
	uv run ruff check src/ tests/ --fix

format: ## Format code with ruff
	uv run ruff format src/ tests/

test: ## Run tests (unit + integration, excludes e2e)
	uv run pytest

test-cov: ## Run tests with coverage report
	uv run pytest --cov=rag_system --cov-report=term-missing

test-e2e: ## Run Playwright E2E browser tests
	uv run pytest tests/e2e/ -m e2e

test-e2e-headed: ## Run Playwright E2E browser tests in browser
	uv run pytest tests/e2e/ -m e2e --headed --slowmo=1000

playwright-install: ## Install Playwright browsers (chromium)
	uv run playwright install chromium

# ── RAG Operations ─────────────────────────────────────────────────

pull-model: ## Pull the default Ollama model (gemma3:1b)
	ollama pull gemma3:1b

ingest: ## Ingest documents from ./documents
	uv run python -m rag_system ingest --folder ./documents

chat: ## Start interactive chat
	uv run python -m rag_system chat

# ── Docker ─────────────────────────────────────────────────────────

docker-up: ## Start all services (Ollama + RAG app)
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

docker-ingest: ## Start services and ingest documents
	docker compose up -d
	docker compose exec rag python -m rag_system ingest --folder ./documents

docker-chat: ## Start chat inside Docker
	docker compose exec rag python -m rag_system chat

# ── Automation ────────────────────────────────────────────────────

backup: ## Back up documents and vector store
	bash scripts/backup.sh

healthcheck: ## Run a single health check
	bash scripts/healthcheck.sh

healthcheck-watch: ## Continuous health monitoring (every 60s)
	bash scripts/healthcheck.sh --watch --log

# ── Utilities ──────────────────────────────────────────────────────

build: ## Build sdist and wheel packages
	uv run python -m build

clean: ## Remove caches and build artifacts
	rm -rf __pycache__ .pytest_cache .ruff_cache chroma_db/ dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
