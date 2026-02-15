#!/usr/bin/env bash
# Container entrypoint — ensure TLS certs exist before starting the app.
set -euo pipefail

if [ ! -f ./certs/cert.pem ] || [ ! -f ./certs/key.pem ]; then
    echo "Certificates not found — generating self-signed certs..."
    bash scripts/generate-certs.sh ./certs
fi

exec "$@"
