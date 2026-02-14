FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv for fast dependency management.
RUN pip install --no-cache-dir uv

# Copy dependency files first (better layer caching).
COPY pyproject.toml README.md ./

# Install dependencies (without the project itself).
RUN uv pip install --system --no-cache-dir .

# Copy source code and cert generation script.
COPY src/ src/
COPY documents/ documents/
COPY scripts/generate-certs.sh scripts/generate-certs.sh

# Install the project.
RUN uv pip install --system --no-cache-dir -e .

# Generate self-signed certs at build time (can be overridden via volume mount).
RUN bash scripts/generate-certs.sh ./certs

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8443

CMD ["uvicorn", "rag_system.web:app", "--host", "0.0.0.0", "--port", "8443", "--ssl-keyfile", "./certs/key.pem", "--ssl-certfile", "./certs/cert.pem"]
