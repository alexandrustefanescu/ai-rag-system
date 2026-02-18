# ── Builder stage ────────────────────────────────────────────────────────────
# Everything installed into a single .venv — no duplicate downloads.
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir uv

# Copy dependency manifests first for better layer caching.
COPY pyproject.toml uv.lock README.md ./

# Install third-party deps only (local package needs src/ which isn't here yet).
# uv creates and manages a single .venv — no manual venv creation needed.
RUN uv sync --frozen --no-install-project --no-dev

# Now add source and install the local package into the same .venv.
COPY src/ src/
RUN uv sync --frozen --no-dev

# ── Runtime stage ─────────────────────────────────────────────────────────────
# Lean image — only the venv, app source, and runtime system packages.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Single apt layer for both runtime dependencies.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openssl gosu \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy the single venv from builder — keep the same path so that
# shebang lines in console scripts (e.g. uvicorn) remain valid.
COPY --from=builder /app/.venv /app/.venv

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy only runtime artefacts; --chown avoids a separate chown -R layer.
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser documents/ documents/
COPY --chown=appuser:appuser scripts/generate-certs.sh scripts/generate-certs.sh
COPY --chown=appuser:appuser scripts/entrypoint.sh scripts/entrypoint.sh

EXPOSE 8443

ENTRYPOINT ["bash", "scripts/entrypoint.sh"]
CMD ["uvicorn", "rag_system.web:app", \
     "--host", "0.0.0.0", \
     "--port", "8443", \
     "--ssl-keyfile", "./certs/key.pem", \
     "--ssl-certfile", "./certs/cert.pem"]
