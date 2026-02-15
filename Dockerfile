FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install openssl for runtime TLS certificate generation.
RUN apt-get update && apt-get install -y --no-install-recommends openssl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management.
RUN pip install --no-cache-dir uv

# Copy dependency files first (better layer caching).
COPY pyproject.toml uv.lock README.md ./

# Install only third-party dependencies (skip building the local package
# since src/rag_system isn't present yet — avoids hatchling build failure).
RUN uv sync --frozen --no-install-project

# Copy source code and scripts.
COPY src/ src/
COPY documents/ documents/
COPY scripts/generate-certs.sh scripts/generate-certs.sh
COPY scripts/entrypoint.sh scripts/entrypoint.sh

# Install the local package now that source is present.
RUN uv pip install --system --no-cache-dir -e .

RUN apt-get update && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

EXPOSE 8443

ENTRYPOINT ["bash", "scripts/entrypoint.sh"]
CMD ["uvicorn", "rag_system.web:app", "--host", "0.0.0.0", "--port", "8443", "--ssl-keyfile", "./certs/key.pem", "--ssl-certfile", "./certs/cert.pem"]
