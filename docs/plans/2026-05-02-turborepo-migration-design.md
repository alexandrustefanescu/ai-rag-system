# Turborepo Migration Design

**Date:** 2026-05-02
**Status:** Approved

## Goal

Restructure the project into a Turborepo monorepo with the existing Python/FastAPI backend as one app and a new Next.js 16 frontend as another. The existing Jinja2/vanilla-JS frontend is replaced entirely — FastAPI becomes API-only.

---

## Repository Structure

```
ai-rag-system/
├── apps/
│   ├── backend/                  ← existing Python code moved here
│   │   ├── src/rag_system/
│   │   ├── tests/
│   │   ├── scripts/
│   │   ├── documents/
│   │   ├── pyproject.toml
│   │   ├── uv.lock
│   │   ├── Makefile
│   │   └── Dockerfile
│   └── frontend/                 ← new Next.js 16 App Router
│       ├── app/
│       ├── components/
│       ├── lib/
│       ├── next.config.ts
│       ├── package.json
│       └── Dockerfile
├── packages/
│   ├── types/                    ← TS interfaces mirroring Pydantic models
│   ├── biome-config/             ← shared biome.json (lint + format)
│   └── tailwind-config/          ← shared Tailwind preset
├── turbo.json
├── pnpm-workspace.yaml
├── package.json                  ← root; devDeps: turbo, biome
├── biome.json                    ← extends @rag/biome-config
├── docker-compose.yml            ← 3 services: ollama, backend, frontend
├── .gitignore
└── CLAUDE.md
```

---

## Docker & Deployment

Three Docker Compose services:

| Service    | Image base          | Port  | Notes                                      |
|------------|---------------------|-------|--------------------------------------------|
| `ollama`   | `ollama/ollama`     | 11434 | Unchanged                                  |
| `backend`  | `python:3.12-slim`  | 8443  | FastAPI, HTTPS, API-only, CORS enabled     |
| `frontend` | `node:20-alpine`    | 3000  | Next.js, multi-stage build, HTTP           |

**Frontend → Backend communication:**
- Docker: `NEXT_PUBLIC_API_URL=http://backend:8443`
- Local dev: `NEXT_PUBLIC_API_URL=https://localhost:8443`

**Frontend Dockerfile:** multi-stage — `node:20-alpine` builder runs `pnpm build`, lean runtime stage serves with `next start`.

---

## Shared Packages

All internal packages use the `@rag/` scope.

### `@rag/types`

Hand-maintained TypeScript interfaces mirroring the FastAPI Pydantic models:

```ts
export type AskRequest      = { question: string; model?: string }
export type AskResponse     = { answer: string; sources: Source[]; tokens: TokenMetrics }
export type Source          = { filename: string; chunk: string; score: number }
export type TokenMetrics    = { prompt: number; completion: number; total: number }
export type Document        = { filename: string; chunks: number; size_bytes: number }
export type Model           = { name: string; size: number; downloaded: boolean }
export type HealthResponse  = { status: string; version: string }
```

### `@rag/biome-config`

Base `biome.json` with lint + format rules. Root `biome.json` extends it. Each app can override locally.

### `@rag/tailwind-config`

Shared Tailwind v4 preset (theme tokens, base plugins). Frontend `tailwind.config.ts` extends it.

---

## Frontend Architecture

**Framework:** Next.js 16, App Router, Turbopack (default bundler).
**Styling:** Tailwind CSS v4, extending `@rag/tailwind-config`.
**State:** React built-ins only (`useState`, `useContext`). No external state library.
**Chat history:** `localStorage` (matches current behavior).
**API layer:** `apps/frontend/lib/api.ts` — thin fetch wrapper typed with `@rag/types`.

### Route Structure

```
app/
├── layout.tsx              ← root layout, providers, global styles
├── page.tsx                ← redirects to /chat
├── chat/
│   ├── page.tsx            ← main chat interface
│   └── [id]/page.tsx       ← individual conversation
├── documents/
│   └── page.tsx            ← document management + upload
└── models/
    └── page.tsx            ← model download/delete panel
```

### Component Structure

```
components/
├── chat/
│   ├── ChatInput.tsx
│   ├── MessageList.tsx
│   ├── StreamingMessage.tsx   ← SSE streaming handler
│   └── SourceCitations.tsx
├── documents/
│   ├── DropZone.tsx
│   └── DocumentTable.tsx
├── models/
│   └── ModelCard.tsx
└── ui/                        ← shared primitives (Button, Badge, etc.)
```

---

## Turborepo Pipeline

`turbo.json` defines tasks with correct dependency ordering:

| Task    | Depends on          | Description                           |
|---------|---------------------|---------------------------------------|
| `build` | `^build`            | Build all packages, then apps         |
| `dev`   | —                   | Start all apps in parallel with watch |
| `lint`  | —                   | Run Biome lint across all JS packages |
| `check` | —                   | Run Biome format check                |

`turbo run dev` starts the FastAPI dev server and `next dev` in parallel.

---

## Backend Changes

- Remove `src/rag_system/templates/` and `src/rag_system/static/`
- Remove Jinja2 from `web.py` — all routes become pure JSON API
- Add CORS middleware to FastAPI (`fastapi.middleware.cors`) allowing the frontend origin
- No other logic changes — RAG engine, vector store, chunker all unchanged

---

## What Does Not Change

- Python tooling: `uv`, `ruff`, `pytest`, `Makefile` targets — all move with the backend
- Ollama and ChromaDB setup — untouched
- `install.sh` / `uninstall.sh` — updated paths only, same user experience
- All existing API endpoints — no breaking changes
