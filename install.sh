#!/usr/bin/env bash
set -euo pipefail

# ── AI RAG System Installer ──────────────────────────────────────────────────
# Works on a fresh device — installs Docker if needed, then sets up everything.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/alexandrustefanescu/ai-rag-system/main/install.sh | bash
#
# Tested on: Raspberry Pi OS, Ubuntu, Debian, Fedora, macOS
# ─────────────────────────────────────────────────────────────────────────────

INSTALL_DIR="${RAG_INSTALL_DIR:-$HOME/ai-rag-system}"
IMAGE="alexandrustefanescu/ai-rag-system:latest"

info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$1"; }
ok()    { printf '\033[1;32m[OK]\033[0m    %s\n' "$1"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$1"; }
err()   { printf '\033[1;31m[ERROR]\033[0m %s\n' "$1" >&2; exit 1; }

# ── Detect OS ────────────────────────────────────────────────────────────────

detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS_ID="${ID:-unknown}"
        OS_ID_LIKE="${ID_LIKE:-}"
    elif [ "$(uname)" = "Darwin" ]; then
        OS_ID="macos"
        OS_ID_LIKE=""
    else
        OS_ID="unknown"
        OS_ID_LIKE=""
    fi
}

is_debian_based() {
    case "$OS_ID" in
        debian|ubuntu|raspbian) return 0 ;;
    esac
    case "$OS_ID_LIKE" in
        *debian*|*ubuntu*) return 0 ;;
    esac
    return 1
}

is_fedora_based() {
    case "$OS_ID" in
        fedora|centos|rhel|rocky|alma) return 0 ;;
    esac
    case "$OS_ID_LIKE" in
        *fedora*|*rhel*) return 0 ;;
    esac
    return 1
}

detect_os
info "Detected OS: $OS_ID"

# ── Prevent running as root (unless inside Docker) ──────────────────────────

if [ ! -f /.dockerenv ]; then
    if [ "$(id -u)" -eq 0 ] || [ -n "${SUDO_USER:-}" ] || [ -n "${SUDO_UID:-}" ]; then
        err "Do not run this script as root or via sudo. Run as a normal user — sudo is used only where needed."
    fi
fi

# ── Install Docker if missing ───────────────────────────────────────────────

install_docker_linux() {
    info "Installing Docker via official install script..."
    curl -fsSL https://get.docker.com | sudo sh

    # Add current user to docker group so sudo is not needed for docker commands.
    if ! groups "$USER" | grep -q docker; then
        sudo usermod -aG docker "$USER"
        warn "Added $USER to the docker group. You may need to log out and back in (or run 'newgrp docker') for this to take effect."
    fi

    # Start and enable Docker service.
    sudo systemctl enable --now docker
}

install_docker_macos() {
    if command -v brew >/dev/null 2>&1; then
        info "Installing Docker Desktop via Homebrew..."
        brew install --cask docker
        info "Opening Docker Desktop — please complete the setup wizard, then re-run this script."
        open /Applications/Docker.app
        exit 0
    else
        err "Docker is not installed. Install Docker Desktop from https://docs.docker.com/desktop/install/mac-install/ then re-run this script."
    fi
}

if ! command -v docker >/dev/null 2>&1; then
    warn "Docker is not installed."

    if [ "$OS_ID" = "macos" ]; then
        install_docker_macos
    elif is_debian_based || is_fedora_based; then
        install_docker_linux
    else
        err "Docker is not installed and auto-install is not supported for $OS_ID. Install Docker manually from https://docs.docker.com/get-docker/ then re-run this script."
    fi
fi

# ── Install Docker Compose plugin if missing ────────────────────────────────

if ! docker compose version >/dev/null 2>&1; then
    info "Installing Docker Compose plugin..."

    if is_debian_based; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq docker-compose-plugin
    elif is_fedora_based; then
        sudo dnf install -y -q docker-compose-plugin
    else
        err "Docker Compose v2 is not installed. Install it from https://docs.docker.com/compose/install/ then re-run this script."
    fi
fi

# ── Verify Docker daemon is running ─────────────────────────────────────────

if ! docker info >/dev/null 2>&1; then
    # Try starting the service (Linux).
    if command -v systemctl >/dev/null 2>&1; then
        info "Docker daemon is not running, starting it..."
        sudo systemctl start docker
        sleep 2
    fi

    # Check again.
    docker info >/dev/null 2>&1 || err "Docker daemon is not running. Start Docker and re-run this script."
fi

ok "Docker is ready ($(docker --version))"

# ── Check available disk space (need ~4 GB for images + models) ─────────────

available_gb=$(df -k "$HOME" | awk 'NR==2 {print int($4/1024/1024)}')
if [ -n "$available_gb" ] && [ "$available_gb" -lt 4 ] 2>/dev/null; then
    warn "Only ${available_gb}GB disk space available. The system needs ~4 GB (images + AI models)."
fi

# ── Check available memory (recommend 4 GB+) ────────────────────────────────

if [ -f /proc/meminfo ]; then
    total_mem_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
    total_mem_gb=$((total_mem_kb / 1024 / 1024))
    if [ "$total_mem_gb" -lt 4 ]; then
        warn "System has ${total_mem_gb}GB RAM. 4GB+ is recommended for running LLM models."
    fi
fi

# ── Check for port conflicts ────────────────────────────────────────────────

check_port() {
    if command -v ss >/dev/null 2>&1; then
        if ss -tlnp 2>/dev/null | grep -q ":$1 "; then
            warn "Port $1 is already in use. The service on that port may conflict."
        fi
    elif command -v lsof >/dev/null 2>&1; then
        if lsof -i :"$1" -sTCP:LISTEN >/dev/null 2>&1; then
            warn "Port $1 is already in use. The service on that port may conflict."
        fi
    fi
}

check_port 8443
check_port 11434

# ── Create project directory ─────────────────────────────────────────────────

if [ -d "$INSTALL_DIR" ]; then
    info "Directory $INSTALL_DIR already exists, updating..."
else
    mkdir -p "$INSTALL_DIR"
    info "Created $INSTALL_DIR"
fi

mkdir -p "$INSTALL_DIR/documents"

# ── Generate self-signed TLS certificates if missing ─────────────────────────

CERT_DIR="$INSTALL_DIR/certs"
mkdir -p "$CERT_DIR"

if [ -f "$CERT_DIR/cert.pem" ] && [ -f "$CERT_DIR/key.pem" ]; then
    ok "TLS certificates already exist in $CERT_DIR"
else
    if ! command -v openssl >/dev/null 2>&1; then
        warn "openssl not found — skipping cert generation. The container entrypoint will generate certs at startup."
    else
        info "Generating self-signed TLS certificates..."
        openssl req -x509 -newkey rsa:2048 -nodes \
            -keyout "$CERT_DIR/key.pem" \
            -out "$CERT_DIR/cert.pem" \
            -days 825 \
            -subj "/CN=localhost" \
            -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" \
            2>/dev/null
        ok "Self-signed certificates created in $CERT_DIR"
    fi
fi

# ── Write docker-compose.yml ─────────────────────────────────────────────────

COMPOSE_FILE="$INSTALL_DIR/docker-compose.yml"

if [ -f "$COMPOSE_FILE" ]; then
    cp "$COMPOSE_FILE" "$COMPOSE_FILE.bak"
    info "Backed up existing docker-compose.yml to docker-compose.yml.bak"
fi

cat > "$COMPOSE_FILE" << 'EOF'
services:
  ollama:
    image: ollama/ollama:latest
    container_name: rag-ollama
    restart: unless-stopped
    ports:
      - "127.0.0.1:11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    healthcheck:
      test: ["CMD", "ollama", "list"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 15s

  ollama-pull:
    image: ollama/ollama:latest
    container_name: rag-ollama-pull
    depends_on:
      ollama:
        condition: service_healthy
    entrypoint: ["/bin/sh", "-c", "ollama pull gemma3:1b && ollama pull llama3.2:1b"]
    environment:
      - OLLAMA_HOST=http://ollama:11434

  rag:
    image: alexandrustefanescu/ai-rag-system:latest
    container_name: rag-app
    restart: unless-stopped
    depends_on:
      ollama:
        condition: service_healthy
      ollama-pull:
        condition: service_completed_successfully
    ports:
      - "127.0.0.1:8443:8443"
    environment:
      - OLLAMA_HOST=http://ollama:11434
    volumes:
      - ./documents:/app/documents
      - chroma_data:/app/chroma_db
      - ./certs:/app/certs

volumes:
  ollama_data:
  chroma_data:
EOF

ok "docker-compose.yml created"

# ── Pull images ──────────────────────────────────────────────────────────────

info "Pulling Docker images (this may take a few minutes)..."
docker pull ollama/ollama:latest
docker pull "$IMAGE"
ok "Images pulled"

# ── Start services ───────────────────────────────────────────────────────────

info "Starting AI RAG System..."
docker compose -f "$COMPOSE_FILE" up -d

ok "AI RAG System is starting!"

echo ""
echo "  ┌─────────────────────────────────────────────────────┐"
echo "  │                                                     │"
echo "  │   AI RAG System installed successfully!             │"
echo "  │                                                     │"
echo "  │   Open:  https://localhost:8443                     │"
echo "  │                                                     │"
echo "  │   Your browser will show a security warning         │"
echo "  │   (self-signed cert) — click Advanced → Proceed.    │"
echo "  │                                                     │"
echo "  │   Add documents to: ~/ai-rag-system/documents/     │"
echo "  │                                                     │"
echo "  │   Commands:                                         │"
echo "  │     cd ~/ai-rag-system                              │"
echo "  │     docker compose logs -f    # view logs           │"
echo "  │     docker compose down       # stop                │"
echo "  │     docker compose up -d      # start again         │"
echo "  │                                                     │"
echo "  └─────────────────────────────────────────────────────┘"
echo ""

info "AI models are downloading in the background (~1-2 GB)."
info "The app will be fully ready in a few minutes."
