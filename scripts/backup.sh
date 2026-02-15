#!/usr/bin/env bash
# Backup documents and ChromaDB vector store to a timestamped archive.
#
# Usage:
#   bash scripts/backup.sh                  # backs up to ./backups/
#   bash scripts/backup.sh /mnt/usb/backups # backs up to a custom directory
#
# Works with both local and Docker setups.
set -euo pipefail

BACKUP_DIR="${1:-./backups}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_NAME="rag-backup-${TIMESTAMP}"
STAGING_DIR="${BACKUP_DIR}/${ARCHIVE_NAME}"

echo "=== RAG System Backup ==="
echo "Timestamp: ${TIMESTAMP}"
echo "Destination: ${BACKUP_DIR}"
echo ""

mkdir -p "${STAGING_DIR}"

# --- Back up documents ---
if [ -d "./documents" ] && [ "$(ls -A ./documents 2>/dev/null)" ]; then
    cp -r ./documents "${STAGING_DIR}/documents"
    DOC_COUNT="$(find "${STAGING_DIR}/documents" -type f | wc -l | tr -d ' ')"
    echo "[OK] Documents: ${DOC_COUNT} file(s) copied"
else
    echo "[SKIP] No documents found in ./documents"
fi

# --- Back up ChromaDB ---
# Check for Docker volume first, then local directory.
if docker compose ps --status running 2>/dev/null | grep -q rag-app; then
    echo "[INFO] Docker detected — exporting chroma_db from rag-app container..."
    if docker compose cp rag-app:/app/chroma_db "${STAGING_DIR}/chroma_db"; then
        echo "[OK] ChromaDB exported from rag-app to ${STAGING_DIR}/chroma_db"
    else
        echo "[WARN] Failed to copy chroma_db from rag-app — falling back to local directory"
    fi
fi

if [ ! -d "${STAGING_DIR}/chroma_db" ]; then
    if [ -d "./chroma_db" ] && [ "$(ls -A ./chroma_db 2>/dev/null)" ]; then
        cp -r ./chroma_db "${STAGING_DIR}/chroma_db"
        echo "[OK] ChromaDB: local directory copied"
    else
        echo "[SKIP] No ChromaDB data found"
    fi
fi

# --- Create compressed archive ---
tar -czf "${BACKUP_DIR}/${ARCHIVE_NAME}.tar.gz" -C "${BACKUP_DIR}" "${ARCHIVE_NAME}"
rm -rf "${STAGING_DIR}"

ARCHIVE_SIZE="$(du -h "${BACKUP_DIR}/${ARCHIVE_NAME}.tar.gz" | cut -f1)"
echo ""
echo "=== Backup complete ==="
echo "Archive: ${BACKUP_DIR}/${ARCHIVE_NAME}.tar.gz (${ARCHIVE_SIZE})"
echo ""
echo "To restore:"
echo "  tar -xzf ${BACKUP_DIR}/${ARCHIVE_NAME}.tar.gz -C ."
echo "  cp -r ${ARCHIVE_NAME}/documents/* ./documents/"
echo "  # Restart the app to pick up the restored data"
