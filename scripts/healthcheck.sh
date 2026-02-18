#!/usr/bin/env bash
# Health monitor for the RAG system.
#
# Checks if the RAG app and Ollama are reachable and logs results.
# Returns exit code 0 if healthy, 1 if any check fails.
#
# Usage:
#   bash scripts/healthcheck.sh                    # single check, prints to stdout
#   bash scripts/healthcheck.sh --log              # single check, appends to log file
#   bash scripts/healthcheck.sh --watch            # continuous monitoring (every 60s)
#   bash scripts/healthcheck.sh --watch --log      # continuous monitoring with logging
#
# Log file: ./logs/health.log (created automatically)
#
# Cron example (check every 5 minutes):
#   */5 * * * * cd /path/to/rag-system && bash scripts/healthcheck.sh --log
set -euo pipefail

RAG_URL="${RAG_URL:-https://localhost:8443}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
LOG_FILE="./logs/health.log"
WATCH_INTERVAL=60
USE_LOG=false
WATCH_MODE=false

for arg in "$@"; do
    case "$arg" in
        --log) USE_LOG=true ;;
        --watch) WATCH_MODE=true ;;
    esac
done

log() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    local message="[${timestamp}] $1"

    if [ "$USE_LOG" = true ]; then
        mkdir -p "$(dirname "$LOG_FILE")"
        echo "$message" >> "$LOG_FILE"
    fi
    echo "$message"
}

check_health() {
    local exit_code=0

    # --- Check RAG app ---
    if curl -sfk "${RAG_URL}/api/v1/health" -o /dev/null --max-time 10 2>/dev/null; then
        HEALTH_JSON="$(curl -sfk "${RAG_URL}/api/v1/health" --max-time 10 2>/dev/null)"
        STATUS="$(echo "$HEALTH_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "unknown")"
        DOC_COUNT="$(echo "$HEALTH_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['documents'])" 2>/dev/null || echo "?")"
        OLLAMA_OK="$(echo "$HEALTH_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['ollama_connected'])" 2>/dev/null || echo "?")"

        if [ "$STATUS" = "healthy" ]; then
            log "OK    rag-app: ${STATUS} | docs: ${DOC_COUNT} | ollama: ${OLLAMA_OK}"
        else
            log "WARN  rag-app: ${STATUS} | docs: ${DOC_COUNT} | ollama: ${OLLAMA_OK}"
            exit_code=1
        fi
    else
        log "FAIL  rag-app: not reachable at ${RAG_URL}"
        exit_code=1
    fi

    # --- Check Ollama directly ---
    if curl -sf "${OLLAMA_URL}/api/tags" -o /dev/null --max-time 10 2>/dev/null; then
        MODEL_COUNT="$(curl -sf "${OLLAMA_URL}/api/tags" --max-time 10 2>/dev/null | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('models',[])))" 2>/dev/null || echo "?")"
        log "OK    ollama: reachable | models: ${MODEL_COUNT}"
    else
        log "FAIL  ollama: not reachable at ${OLLAMA_URL}"
        exit_code=1
    fi

    # --- Check Docker containers (if Docker is available) ---
    if command -v docker &>/dev/null && docker compose ps &>/dev/null 2>&1; then
        RUNNING="$(docker compose ps --status running --format '{{.Name}}' 2>/dev/null | wc -l | tr -d ' ')"
        TOTAL="$(docker compose ps --format '{{.Name}}' 2>/dev/null | wc -l | tr -d ' ')"
        if [ "$RUNNING" = "$TOTAL" ] && [ "$TOTAL" != "0" ]; then
            log "OK    docker: ${RUNNING}/${TOTAL} containers running"
        else
            log "WARN  docker: ${RUNNING}/${TOTAL} containers running"
            exit_code=1
        fi
    fi

    return $exit_code
}

if [ "$WATCH_MODE" = true ]; then
    log "--- Health monitor started (interval: ${WATCH_INTERVAL}s) ---"
    while true; do
        check_health || true
        sleep "$WATCH_INTERVAL"
    done
else
    check_health
fi
