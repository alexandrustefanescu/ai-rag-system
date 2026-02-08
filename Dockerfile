FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv for fast dependency management.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (better layer caching).
COPY pyproject.toml ./

# Install dependencies (without the project itself).
RUN uv pip install --system --no-cache-dir .

# Copy source code.
COPY src/ src/
COPY documents/ documents/

# Install the project.
RUN uv pip install --system --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "rag_system.web:app", "--host", "0.0.0.0", "--port", "8000"]
