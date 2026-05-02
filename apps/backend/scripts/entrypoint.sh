#!/usr/bin/env bash
# Container entrypoint — ensure TLS certs exist, then drop to appuser.
set -euo pipefail

# Generate self-signed certificates if not already present.
# Runs as root so it can write to the bind-mounted ./certs directory.
if [ ! -f ./certs/cert.pem ] || [ ! -f ./certs/key.pem ]; then
    echo "Certificates not found — generating self-signed certs..."
    bash scripts/generate-certs.sh ./certs
fi

# Ensure appuser can read the certificates.
chown -R appuser:appuser ./certs 2>/dev/null || true

# Ensure appuser owns the ChromaDB data directory.
mkdir -p ./chroma_db
chown -R appuser:appuser ./chroma_db

# Drop privileges and exec the main process as appuser.
exec gosu appuser "$@"
