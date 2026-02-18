#!/usr/bin/env bash
set -euo pipefail

# ── AI RAG System Uninstaller ────────────────────────────────────────────────
# Removes containers, images, volumes, and the install directory.
#
# Usage:
#   bash -c "$(curl -fsSL https://raw.githubusercontent.com/alexandrustefanescu/ai-rag-system/main/uninstall.sh)"
#
# ─────────────────────────────────────────────────────────────────────────────

INSTALL_DIR="${RAG_INSTALL_DIR:-$HOME/ai-rag-system}"

info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$1"; }
ok()    { printf '\033[1;32m[OK]\033[0m    %s\n' "$1"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$1"; }
err()   { printf '\033[1;31m[ERROR]\033[0m %s\n' "$1" >&2; exit 1; }

# ── Docker command helper ────────────────────────────────────────────────────

run_docker() {
    if docker info >/dev/null 2>&1; then
        docker "$@"
    else
        sudo docker "$@"
    fi
}

# ── Confirm ──────────────────────────────────────────────────────────────────

echo ""
echo "  ┌─────────────────────────────────────────────────────┐"
echo "  │                                                     │"
echo "  │   AI RAG System Uninstaller                         │"
echo "  │                                                     │"
echo "  │   This will remove:                                 │"
echo "  │     - Docker containers (rag-app, rag-ollama)       │"
echo "  │     - Docker images (rag-system, ollama)            │"
echo "  │     - Docker volumes (chroma_data, ollama_data)     │"
echo "  │     - Install directory: $INSTALL_DIR"
echo "  │                                                     │"
echo "  │                                                     │"
echo "  └─────────────────────────────────────────────────────┘"
echo ""

read -rp "Are you sure you want to uninstall? [y/N] " confirm </dev/tty
case "$confirm" in
    [yY]|[yY][eE][sS]) ;;
    *) info "Uninstall cancelled."; exit 0 ;;
esac

# ── Stop and remove containers ───────────────────────────────────────────────

COMPOSE_FILE="$INSTALL_DIR/docker-compose.yml"

if [ -f "$COMPOSE_FILE" ]; then
    info "Stopping containers..."
    run_docker compose -f "$COMPOSE_FILE" down --timeout 10 2>/dev/null || true
    ok "Containers stopped"
else
    # Containers might exist without a compose file.
    for name in rag-app rag-ollama; do
        if run_docker ps -aq -f "name=$name" | grep -q .; then
            info "Removing container $name..."
            run_docker rm -f "$name" 2>/dev/null || true
        fi
    done
fi

# ── Remove Docker volumes ───────────────────────────────────────────────────

for vol in ai-rag-system_chroma_data ai-rag-system_ollama_data chroma_data ollama_data; do
    if run_docker volume ls -q | grep -qx "$vol"; then
        info "Removing volume $vol..."
        run_docker volume rm "$vol" 2>/dev/null || true
    fi
done
ok "Volumes removed"

# ── Remove Docker images ────────────────────────────────────────────────────

read -rp "Remove Docker images (ollama, rag-system)? [y/N] " rm_images </dev/tty
case "$rm_images" in
    [yY]|[yY][eE][sS])
        for img in alexandrustefanescu/ai-rag-system:latest ollama/ollama:latest; do
            if run_docker images -q "$img" 2>/dev/null | grep -q .; then
                info "Removing image $img..."
                run_docker rmi "$img" 2>/dev/null || true
            fi
        done
        ok "Images removed"
        ;;
    *)
        info "Keeping Docker images."
        ;;
esac

# ── Remove install directory ─────────────────────────────────────────────────

if [ -d "$INSTALL_DIR" ]; then
    read -rp "Remove $INSTALL_DIR and all its contents? [y/N] " rm_dir </dev/tty
    case "$rm_dir" in
        [yY]|[yY][eE][sS])
            rm -rf "$INSTALL_DIR"
            ok "Removed $INSTALL_DIR"
            ;;
        *)
            info "Keeping $INSTALL_DIR."
            ;;
    esac
else
    info "Install directory $INSTALL_DIR not found — skipping."
fi

# ── Done ─────────────────────────────────────────────────────────────────────

echo ""
ok "AI RAG System has been uninstalled."
echo ""
