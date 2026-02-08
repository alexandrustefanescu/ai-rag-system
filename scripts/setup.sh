#!/usr/bin/env bash
# Quick setup script for the RAG system.
set -euo pipefail

echo "🔧 Setting up RAG System..."

# Check for uv
if ! command -v uv &>/dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create venv and install
echo "📦 Installing dependencies..."
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Check for Ollama
if command -v ollama &>/dev/null; then
    echo "🤖 Pulling tinyllama model..."
    ollama pull tinyllama
else
    echo "⚠️  Ollama not found. Install it from https://ollama.com"
    echo "   Then run: ollama pull tinyllama"
fi

echo ""
echo "✅ Setup complete! Next steps:"
echo "   source .venv/bin/activate"
echo "   make ingest     # Ingest sample documents"
echo "   make chat       # Start chatting"
