# Turborepo Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the repo into a Turborepo monorepo with the Python/FastAPI backend in `apps/backend/`, a new Next.js 16 App Router frontend in `apps/frontend/`, and three shared packages under `packages/`.

**Architecture:** pnpm workspaces + Turborepo orchestrate all JS packages; a minimal `package.json` in `apps/backend/` makes the Python server a Turborepo-managed task. The three services (ollama, backend, frontend) run via a single `docker-compose.yml` at the repo root. The existing Jinja2/static UI is deleted — FastAPI becomes API-only with CORS.

**Tech Stack:** Python 3.12 / FastAPI / uv, Next.js 16 / App Router / Turbopack, Tailwind CSS v4, Biome, pnpm, Turborepo, Docker multi-stage builds.

---

## Task 1: Move backend files into apps/backend/

**Files:**
- Create: `apps/backend/` (directory)
- Move: everything at root except `.git`, `.gitignore`, `CLAUDE.md`, `README.md`, `docker-compose.yml`, `install.sh`, `uninstall.sh`, `docs/`

**Step 1: Create directory structure**

```bash
mkdir -p apps/backend
```

**Step 2: Move Python project files with git mv**

```bash
git mv src apps/backend/src
git mv tests apps/backend/tests
git mv scripts apps/backend/scripts
git mv documents apps/backend/documents
git mv pyproject.toml apps/backend/pyproject.toml
git mv uv.lock apps/backend/uv.lock
git mv Makefile apps/backend/Makefile
git mv Dockerfile apps/backend/Dockerfile
git mv .dockerignore apps/backend/.dockerignore
git mv .python-version apps/backend/.python-version
```

**Step 3: Verify the move**

```bash
ls apps/backend/
# Expected: Dockerfile  Makefile  documents  pyproject.toml  scripts  src  tests  uv.lock
```

**Step 4: Update .gitignore at root — add JS-specific ignores**

Append to `.gitignore`:
```
node_modules/
.next/
.turbo/
pnpm-lock.yaml
*.tsbuildinfo
```

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: move backend into apps/backend"
```

---

## Task 2: Add backend package.json for Turborepo

**Files:**
- Create: `apps/backend/package.json`

This gives the Python backend a Turborepo-managed `dev` task so `turbo run dev` starts both servers.

**Step 1: Create apps/backend/package.json**

```json
{
  "name": "backend",
  "private": true,
  "scripts": {
    "dev": "bash scripts/generate-certs.sh && uv run uvicorn rag_system.web:app --reload --host 0.0.0.0 --port 8443 --ssl-keyfile ./certs/key.pem --ssl-certfile ./certs/cert.pem",
    "lint": "uv run ruff check src/ tests/ --fix",
    "format": "uv run ruff format src/ tests/",
    "test": "uv run pytest"
  }
}
```

**Step 2: Commit**

```bash
git add apps/backend/package.json
git commit -m "chore: add backend package.json for turborepo tasks"
```

---

## Task 3: Set up Turborepo root config

**Files:**
- Create: `package.json` (root)
- Create: `pnpm-workspace.yaml`
- Create: `turbo.json`

**Step 1: Check pnpm is installed**

```bash
pnpm --version
# If not installed: npm install -g pnpm
```

**Step 2: Create pnpm-workspace.yaml**

```yaml
packages:
  - "apps/*"
  - "packages/*"
```

**Step 3: Create root package.json**

```json
{
  "name": "ai-rag-system",
  "private": true,
  "scripts": {
    "build": "turbo run build",
    "dev": "turbo run dev",
    "lint": "turbo run lint",
    "check": "biome check ."
  },
  "devDependencies": {
    "@biomejs/biome": "^1.9.0",
    "turbo": "^2.0.0"
  },
  "packageManager": "pnpm@10.0.0"
}
```

**Step 4: Create turbo.json**

```json
{
  "$schema": "https://turbo.build/schema.json",
  "tasks": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": [".next/**", "!.next/cache/**", "dist/**"]
    },
    "dev": {
      "cache": false,
      "persistent": true
    },
    "lint": {
      "dependsOn": ["^lint"]
    }
  }
}
```

**Step 5: Install root dependencies**

```bash
pnpm install
```

**Step 6: Verify turbo is available**

```bash
pnpm turbo --version
# Expected: 2.x.x
```

**Step 7: Commit**

```bash
git add package.json pnpm-workspace.yaml turbo.json pnpm-lock.yaml
git commit -m "chore: set up turborepo root with pnpm workspaces"
```

---

## Task 4: Create @rag/biome-config package

**Files:**
- Create: `packages/biome-config/package.json`
- Create: `packages/biome-config/biome.json`
- Create: `biome.json` (root, extends the package)

**Step 1: Create packages/biome-config/package.json**

```json
{
  "name": "@rag/biome-config",
  "version": "0.0.1",
  "private": true,
  "exports": {
    ".": "./biome.json"
  }
}
```

**Step 2: Create packages/biome-config/biome.json**

```json
{
  "$schema": "https://biomejs.dev/schemas/1.9.4/schema.json",
  "organizeImports": { "enabled": true },
  "linter": {
    "enabled": true,
    "rules": {
      "recommended": true
    }
  },
  "formatter": {
    "enabled": true,
    "indentStyle": "space",
    "indentWidth": 2,
    "lineWidth": 100
  },
  "javascript": {
    "formatter": {
      "quoteStyle": "double",
      "trailingCommas": "all",
      "semicolons": "always"
    }
  }
}
```

**Step 3: Create root biome.json**

```json
{
  "$schema": "https://biomejs.dev/schemas/1.9.4/schema.json",
  "extends": ["@rag/biome-config"]
}
```

**Step 4: Run pnpm install to link the package**

```bash
pnpm install
```

**Step 5: Verify biome picks up the config**

```bash
pnpm biome check --help
# Expected: exits cleanly
```

**Step 6: Commit**

```bash
git add packages/biome-config/ biome.json pnpm-lock.yaml
git commit -m "chore: add @rag/biome-config shared biome config"
```

---

## Task 5: Create @rag/types package

**Files:**
- Create: `packages/types/package.json`
- Create: `packages/types/src/index.ts`
- Create: `packages/types/tsconfig.json`

**Step 1: Create packages/types/package.json**

```json
{
  "name": "@rag/types",
  "version": "0.0.1",
  "private": true,
  "exports": {
    ".": {
      "types": "./src/index.ts",
      "default": "./src/index.ts"
    }
  },
  "devDependencies": {
    "typescript": "^5.0.0"
  }
}
```

**Step 2: Create packages/types/tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "declaration": true,
    "isolatedModules": true
  },
  "include": ["src"]
}
```

**Step 3: Create packages/types/src/index.ts**

These mirror the Pydantic models in `apps/backend/src/rag_system/web.py` exactly.

```typescript
export type AskRequest = {
  question: string;
  model?: string;
};

export type SourceResponse = {
  text: string;
  source: string;
  relevance: number;
};

export type GenerationMetrics = {
  duration_s: number;
  tokens_generated: number;
  tokens_per_second: number;
};

export type AskResponse = {
  answer: string;
  sources: SourceResponse[];
  metrics: GenerationMetrics | null;
};

export type IngestResponse = {
  status: string;
  chunks: number;
};

export type UploadResponse = {
  status: string;
  files_saved: number;
  chunks: number;
};

export type DocumentInfo = {
  filename: string;
  size_kb: number;
  chunk_count: number;
};

export type DocumentListResponse = {
  files: DocumentInfo[];
};

export type DeleteResponse = {
  status: string;
  chunks: number;
};

export type HealthResponse = {
  status: string;
  ollama_connected: boolean;
  documents: number;
};

export type StatusResponse = {
  documents: number;
  model: string;
  available_models: string[];
  downloaded_models: string[];
  ollama_connected: boolean;
};

export type ModelInfo = {
  name: string;
  size_mb: number;
  downloaded: boolean;
};

export type ModelListResponse = {
  models: ModelInfo[];
};

export type PullRequest = {
  model: string;
};

export type PullResponse = {
  status: string;
};

export type PullStatusResponse = {
  status: string;
  progress: string;
};

export type DeleteModelResponse = {
  status: string;
};
```

**Step 4: Run pnpm install to link the package**

```bash
pnpm install
```

**Step 5: Commit**

```bash
git add packages/types/ pnpm-lock.yaml
git commit -m "chore: add @rag/types shared TypeScript types"
```

---

## Task 6: Create @rag/tailwind-config package

**Files:**
- Create: `packages/tailwind-config/package.json`
- Create: `packages/tailwind-config/theme.css`

Tailwind v4 uses CSS-first config (`@theme`), so this package exports a CSS file.

**Step 1: Create packages/tailwind-config/package.json**

```json
{
  "name": "@rag/tailwind-config",
  "version": "0.0.1",
  "private": true,
  "exports": {
    "./theme.css": "./theme.css"
  }
}
```

**Step 2: Create packages/tailwind-config/theme.css**

```css
@theme {
  --font-sans: "Inter", ui-sans-serif, system-ui, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, monospace;

  --color-brand-50: oklch(97% 0.01 262);
  --color-brand-100: oklch(94% 0.03 262);
  --color-brand-200: oklch(88% 0.06 262);
  --color-brand-300: oklch(80% 0.10 262);
  --color-brand-400: oklch(70% 0.15 262);
  --color-brand-500: oklch(58% 0.20 262);
  --color-brand-600: oklch(50% 0.20 262);
  --color-brand-700: oklch(42% 0.18 262);
  --color-brand-800: oklch(34% 0.14 262);
  --color-brand-900: oklch(26% 0.10 262);
  --color-brand-950: oklch(18% 0.07 262);

  --radius-sm: 0.375rem;
  --radius-md: 0.5rem;
  --radius-lg: 0.75rem;
  --radius-xl: 1rem;
}
```

**Step 3: Run pnpm install**

```bash
pnpm install
```

**Step 4: Commit**

```bash
git add packages/tailwind-config/ pnpm-lock.yaml
git commit -m "chore: add @rag/tailwind-config shared Tailwind v4 theme"
```

---

## Task 7: Strip backend of templates/static, add CORS

**Files:**
- Delete: `apps/backend/src/rag_system/templates/`
- Delete: `apps/backend/src/rag_system/static/`
- Modify: `apps/backend/src/rag_system/web.py`
- Modify: `apps/backend/src/rag_system/config.py`
- Modify: `apps/backend/pyproject.toml`
- Test: `apps/backend/tests/test_web.py` (existing)

**Step 1: Write a failing test for CORS headers**

Add to `apps/backend/tests/test_web.py` (or create it if it doesn't exist):

```python
def test_cors_allows_frontend_origin(client):
    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert (
        response.headers.get("access-control-allow-origin") == "http://localhost:3000"
    )


def test_root_returns_404_not_html(client):
    response = client.get("/")
    assert response.status_code == 404
```

**Step 2: Run tests to confirm they fail**

```bash
cd apps/backend && uv run pytest tests/test_web.py::test_cors_allows_frontend_origin tests/test_web.py::test_root_returns_404_not_html -v
# Expected: FAIL — CORS headers absent, root serves HTML
```

**Step 3: Add CORS config to config.py**

In `apps/backend/src/rag_system/config.py`, add a `cors_origins` field to `AppConfig`:

```python
from pydantic import Field, field_validator, model_validator
# existing imports ...

class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(frozen=True)

    documents_dir: str = "./documents"
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://frontend:3000"]
    )
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ssl: SSLConfig = Field(default_factory=SSLConfig)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_origins(cls, v: object) -> list[str]:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v  # type: ignore[return-value]
```

**Step 4: Update web.py**

Remove these lines from `apps/backend/src/rag_system/web.py`:
```python
from fastapi.staticfiles import StaticFiles
# ...
TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
# ...
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
# ...
app.mount("/", StaticFiles(directory=str(TEMPLATES_DIR), html=True), name="ui")
```

Add after `app = FastAPI(...)`:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=_config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Also remove the `Path` import if it's no longer used elsewhere (it is still used in route handlers, so keep it).

**Step 5: Remove jinja2 from pyproject.toml**

In `apps/backend/pyproject.toml`, remove the line:
```
"jinja2>=3.1.0",
```

**Step 6: Delete templates and static directories**

```bash
git rm -r apps/backend/src/rag_system/templates/
git rm -r apps/backend/src/rag_system/static/
```

**Step 7: Run tests to confirm they pass**

```bash
cd apps/backend && uv run pytest tests/test_web.py::test_cors_allows_frontend_origin tests/test_web.py::test_root_returns_404_not_html -v
# Expected: PASS
```

**Step 8: Run full test suite to check for regressions**

```bash
cd apps/backend && uv run pytest -v
# Expected: all tests pass
```

**Step 9: Run format + lint**

```bash
cd apps/backend && uv run ruff format src/ tests/ && uv run ruff check src/ tests/ --fix
```

**Step 10: Commit**

```bash
git add apps/backend/
git commit -m "feat: strip backend UI, add CORS middleware for Next.js frontend"
```

---

## Task 8: Scaffold Next.js 16 frontend

**Files:**
- Create: `apps/frontend/` (full Next.js project)

**Step 1: Scaffold from apps/ directory**

```bash
cd apps
pnpm create next-app@latest frontend \
  --typescript \
  --tailwind \
  --app \
  --no-eslint \
  --no-src-dir \
  --no-import-alias
```

When prompted, accept all defaults. This creates `apps/frontend/`.

**Step 2: Verify the scaffold**

```bash
ls apps/frontend/
# Expected: app/ components/ public/ package.json next.config.ts tsconfig.json
```

**Step 3: Update apps/frontend/package.json — add name and workspace deps**

Replace the generated `package.json` with:

```json
{
  "name": "frontend",
  "version": "0.0.1",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "biome check ."
  },
  "dependencies": {
    "@rag/types": "workspace:*",
    "next": "^16.0.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0"
  },
  "devDependencies": {
    "@rag/biome-config": "workspace:*",
    "@rag/tailwind-config": "workspace:*",
    "@types/node": "^22.0.0",
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "tailwindcss": "^4.0.0",
    "typescript": "^5.0.0"
  }
}
```

**Step 4: Update apps/frontend/next.config.ts**

```typescript
import path from "path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  outputFileTracingRoot: path.join(__dirname, "../../"),
  turbopack: {},
};

export default nextConfig;
```

**Step 5: Update apps/frontend/tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2017",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": {
      "@/*": ["./*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

**Step 6: Add biome.json in apps/frontend/**

```json
{
  "$schema": "https://biomejs.dev/schemas/1.9.4/schema.json",
  "extends": ["@rag/biome-config"],
  "files": {
    "ignore": [".next/**", "node_modules/**"]
  }
}
```

**Step 7: Update apps/frontend/app/globals.css — switch to Tailwind v4 imports**

Replace the generated file contents with:

```css
@import "tailwindcss";
@import "@rag/tailwind-config/theme.css";

* {
  box-sizing: border-box;
}
```

**Step 8: Install all workspace dependencies from root**

```bash
cd ../..   # back to repo root
pnpm install
```

**Step 9: Verify Next.js builds**

```bash
pnpm --filter frontend build
# Expected: build succeeds, no TypeScript errors
```

**Step 10: Commit**

```bash
git add apps/frontend/ pnpm-lock.yaml
git commit -m "feat: scaffold Next.js 16 App Router frontend"
```

---

## Task 9: Implement the API layer

**Files:**
- Create: `apps/frontend/lib/api.ts`
- Create: `apps/frontend/lib/env.ts`

**Step 1: Create apps/frontend/lib/env.ts**

```typescript
export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "https://localhost:8443";
```

**Step 2: Create apps/frontend/lib/api.ts**

```typescript
import type {
  AskRequest,
  AskResponse,
  DeleteModelResponse,
  DeleteResponse,
  DocumentListResponse,
  HealthResponse,
  IngestResponse,
  ModelListResponse,
  PullRequest,
  PullResponse,
  PullStatusResponse,
  StatusResponse,
  UploadResponse,
} from "@rag/types";

import { API_URL } from "./env";

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(`${API_URL}/api/v1${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export async function getStatus(): Promise<StatusResponse> {
  return request<StatusResponse>("/status");
}

export async function ask(body: AskRequest): Promise<AskResponse> {
  return request<AskResponse>("/ask", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function askStream(body: AskRequest): EventSource {
  const url = new URL(`${API_URL}/api/v1/ask/stream`);
  // EventSource doesn't support POST; use fetch with ReadableStream instead.
  // Return a plain EventSource-compatible object via a POST fetch.
  throw new Error(
    "Use streamAsk() for streaming. EventSource only supports GET.",
  );
}

export async function* streamAsk(
  body: AskRequest,
  signal?: AbortSignal,
): AsyncGenerator<string> {
  const res = await fetch(`${API_URL}/api/v1/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${text}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        yield line.slice(6);
      }
    }
  }
}

export async function listDocuments(): Promise<DocumentListResponse> {
  return request<DocumentListResponse>("/documents");
}

export async function uploadFiles(files: File[]): Promise<UploadResponse> {
  const form = new FormData();
  for (const file of files) form.append("files", file);
  const res = await fetch(`${API_URL}/api/v1/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json() as Promise<UploadResponse>;
}

export async function deleteDocument(filename: string): Promise<DeleteResponse> {
  return request<DeleteResponse>(`/documents/${encodeURIComponent(filename)}`, {
    method: "DELETE",
  });
}

export async function ingest(): Promise<IngestResponse> {
  return request<IngestResponse>("/ingest", { method: "POST" });
}

export async function listModels(): Promise<ModelListResponse> {
  return request<ModelListResponse>("/models");
}

export async function pullModel(body: PullRequest): Promise<PullResponse> {
  return request<PullResponse>("/models/pull", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function getModelStatus(
  modelName: string,
): Promise<PullStatusResponse> {
  return request<PullStatusResponse>(
    `/models/${encodeURIComponent(modelName)}/status`,
  );
}

export async function deleteModel(
  modelName: string,
): Promise<DeleteModelResponse> {
  return request<DeleteModelResponse>(
    `/models/${encodeURIComponent(modelName)}`,
    { method: "DELETE" },
  );
}
```

**Step 3: Verify TypeScript compiles cleanly**

```bash
pnpm --filter frontend build
# Expected: no TypeScript errors
```

**Step 4: Commit**

```bash
git add apps/frontend/lib/
git commit -m "feat: add typed API layer for backend endpoints"
```

---

## Task 10: Implement root layout and navigation

**Files:**
- Modify: `apps/frontend/app/layout.tsx`
- Modify: `apps/frontend/app/page.tsx`
- Create: `apps/frontend/components/nav/Sidebar.tsx`

**Step 1: Update apps/frontend/app/layout.tsx**

```tsx
import type { Metadata } from "next";
import { Sidebar } from "@/components/nav/Sidebar";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI RAG System",
  description: "Local Retrieval-Augmented Generation powered by Ollama",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="flex h-screen bg-gray-950 text-gray-100 antialiased">
        <Sidebar />
        <main className="flex-1 overflow-auto">{children}</main>
      </body>
    </html>
  );
}
```

**Step 2: Update apps/frontend/app/page.tsx — redirect to /chat**

```tsx
import { redirect } from "next/navigation";

export default function Home() {
  redirect("/chat");
}
```

**Step 3: Create apps/frontend/components/nav/Sidebar.tsx**

```tsx
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/chat", label: "Chat", icon: "💬" },
  { href: "/documents", label: "Documents", icon: "📄" },
  { href: "/models", label: "Models", icon: "🤖" },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <nav className="flex w-52 flex-col border-r border-gray-800 bg-gray-900 p-4">
      <div className="mb-8">
        <h1 className="text-sm font-semibold tracking-widest text-gray-400 uppercase">
          RAG System
        </h1>
      </div>
      <ul className="space-y-1">
        {links.map(({ href, label, icon }) => (
          <li key={href}>
            <Link
              href={href}
              className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors ${
                pathname.startsWith(href)
                  ? "bg-brand-500/20 text-brand-300"
                  : "text-gray-400 hover:bg-gray-800 hover:text-gray-100"
              }`}
            >
              <span>{icon}</span>
              {label}
            </Link>
          </li>
        ))}
      </ul>
    </nav>
  );
}
```

**Step 4: Verify build**

```bash
pnpm --filter frontend build
# Expected: builds cleanly
```

**Step 5: Commit**

```bash
git add apps/frontend/app/layout.tsx apps/frontend/app/page.tsx apps/frontend/components/nav/
git commit -m "feat: add root layout and sidebar navigation"
```

---

## Task 11: Implement chat page

**Files:**
- Create: `apps/frontend/app/chat/page.tsx`
- Create: `apps/frontend/components/chat/ChatInput.tsx`
- Create: `apps/frontend/components/chat/MessageList.tsx`
- Create: `apps/frontend/components/chat/SourceCitations.tsx`

**Step 1: Create apps/frontend/components/chat/SourceCitations.tsx**

```tsx
import type { SourceResponse } from "@rag/types";

export function SourceCitations({ sources }: { sources: SourceResponse[] }) {
  if (!sources.length) return null;

  return (
    <div className="mt-3 space-y-2">
      <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">
        Sources
      </p>
      {sources.map((src, i) => (
        <div
          key={i}
          className="rounded-md border border-gray-700 bg-gray-800/50 p-3 text-xs"
        >
          <p className="mb-1 font-medium text-brand-400">{src.source}</p>
          <p className="text-gray-400 line-clamp-3">{src.text}</p>
          <p className="mt-1 text-gray-600">
            relevance: {(src.relevance * 100).toFixed(0)}%
          </p>
        </div>
      ))}
    </div>
  );
}
```

**Step 2: Create apps/frontend/components/chat/MessageList.tsx**

```tsx
import type { SourceResponse } from "@rag/types";
import { SourceCitations } from "./SourceCitations";

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceResponse[];
  streaming?: boolean;
};

export function MessageList({ messages }: { messages: Message[] }) {
  return (
    <div className="flex flex-col gap-6 p-6">
      {messages.map((msg) => (
        <div
          key={msg.id}
          className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
        >
          <div
            className={`max-w-2xl rounded-2xl px-4 py-3 text-sm ${
              msg.role === "user"
                ? "bg-brand-600 text-white"
                : "bg-gray-800 text-gray-100"
            }`}
          >
            <p className="whitespace-pre-wrap">
              {msg.content}
              {msg.streaming && (
                <span className="ml-1 inline-block h-4 w-0.5 animate-pulse bg-brand-400" />
              )}
            </p>
            {msg.role === "assistant" && msg.sources && (
              <SourceCitations sources={msg.sources} />
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
```

**Step 3: Create apps/frontend/components/chat/ChatInput.tsx**

```tsx
"use client";

import { useRef } from "react";

export function ChatInput({
  onSubmit,
  disabled,
}: {
  onSubmit: (question: string) => void;
  disabled: boolean;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  function submit() {
    const value = ref.current?.value.trim();
    if (!value || disabled) return;
    onSubmit(value);
    if (ref.current) ref.current.value = "";
  }

  return (
    <div className="border-t border-gray-800 p-4">
      <div className="flex items-end gap-3 rounded-xl border border-gray-700 bg-gray-800 px-4 py-3">
        <textarea
          ref={ref}
          rows={1}
          placeholder="Ask a question about your documents…"
          disabled={disabled}
          onKeyDown={handleKeyDown}
          className="flex-1 resize-none bg-transparent text-sm text-gray-100 placeholder-gray-500 outline-none disabled:opacity-50"
        />
        <button
          type="button"
          onClick={submit}
          disabled={disabled}
          className="rounded-lg bg-brand-500 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-brand-600 disabled:opacity-40"
        >
          Send
        </button>
      </div>
      <p className="mt-2 text-center text-xs text-gray-600">
        Enter to send · Shift+Enter for new line
      </p>
    </div>
  );
}
```

**Step 4: Create apps/frontend/app/chat/page.tsx**

```tsx
"use client";

import { useRef, useState } from "react";
import { ChatInput } from "@/components/chat/ChatInput";
import { MessageList, type Message } from "@/components/chat/MessageList";
import { streamAsk } from "@/lib/api";

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  async function handleSubmit(question: string) {
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: question,
    };
    const assistantId = crypto.randomUUID();
    const assistantMsg: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      streaming: true,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      let fullContent = "";
      for await (const chunk of streamAsk({ question }, controller.signal)) {
        fullContent += chunk;
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, content: fullContent } : m,
          ),
        );
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content: `Error: ${(err as Error).message}`,
                  streaming: false,
                }
              : m,
          ),
        );
      }
    } finally {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId ? { ...m, streaming: false } : m,
        ),
      );
      setStreaming(false);
    }
  }

  return (
    <div className="flex h-full flex-col">
      <header className="border-b border-gray-800 px-6 py-4">
        <h2 className="text-lg font-semibold">Chat</h2>
      </header>
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <div className="flex h-full items-center justify-center text-gray-500 text-sm">
            Upload documents, then ask a question.
          </div>
        ) : (
          <MessageList messages={messages} />
        )}
      </div>
      <ChatInput onSubmit={handleSubmit} disabled={streaming} />
    </div>
  );
}
```

**Step 5: Verify build**

```bash
pnpm --filter frontend build
# Expected: builds cleanly
```

**Step 6: Commit**

```bash
git add apps/frontend/app/chat/ apps/frontend/components/chat/
git commit -m "feat: implement streaming chat page"
```

---

## Task 12: Implement documents page

**Files:**
- Create: `apps/frontend/app/documents/page.tsx`
- Create: `apps/frontend/components/documents/DropZone.tsx`
- Create: `apps/frontend/components/documents/DocumentTable.tsx`

**Step 1: Create apps/frontend/components/documents/DocumentTable.tsx**

```tsx
import type { DocumentInfo } from "@rag/types";

export function DocumentTable({
  documents,
  onDelete,
}: {
  documents: DocumentInfo[];
  onDelete: (filename: string) => void;
}) {
  if (!documents.length) {
    return (
      <p className="py-8 text-center text-sm text-gray-500">
        No documents indexed yet.
      </p>
    );
  }

  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b border-gray-800 text-left text-xs text-gray-500 uppercase tracking-wide">
          <th className="pb-2 font-medium">File</th>
          <th className="pb-2 font-medium">Size</th>
          <th className="pb-2 font-medium">Chunks</th>
          <th className="pb-2 font-medium" />
        </tr>
      </thead>
      <tbody>
        {documents.map((doc) => (
          <tr key={doc.filename} className="border-b border-gray-800/50">
            <td className="py-3 text-gray-200">{doc.filename}</td>
            <td className="py-3 text-gray-400">{doc.size_kb} KB</td>
            <td className="py-3 text-gray-400">{doc.chunk_count}</td>
            <td className="py-3 text-right">
              <button
                type="button"
                onClick={() => onDelete(doc.filename)}
                className="rounded px-2 py-1 text-xs text-red-400 hover:bg-red-900/30 transition"
              >
                Delete
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

**Step 2: Create apps/frontend/components/documents/DropZone.tsx**

```tsx
"use client";

import { useRef, useState } from "react";

export function DropZone({
  onFiles,
  uploading,
}: {
  onFiles: (files: File[]) => void;
  uploading: boolean;
}) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length) onFiles(files);
  }

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onClick={() => inputRef.current?.click()}
      className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-10 text-sm transition ${
        dragging
          ? "border-brand-400 bg-brand-500/10"
          : "border-gray-700 hover:border-gray-500"
      } ${uploading ? "pointer-events-none opacity-50" : ""}`}
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".txt,.md,.pdf"
        className="hidden"
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          if (files.length) onFiles(files);
        }}
      />
      <span className="text-2xl">📂</span>
      <p className="text-gray-400">
        {uploading ? "Uploading…" : "Drop files or click to upload"}
      </p>
      <p className="text-xs text-gray-600">Supported: .txt .md .pdf (max 50 MB each)</p>
    </div>
  );
}
```

**Step 3: Create apps/frontend/app/documents/page.tsx**

```tsx
"use client";

import { useCallback, useEffect, useState } from "react";
import { DocumentTable } from "@/components/documents/DocumentTable";
import { DropZone } from "@/components/documents/DropZone";
import { deleteDocument, listDocuments, uploadFiles } from "@/lib/api";
import type { DocumentInfo } from "@rag/types";

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchDocs = useCallback(async () => {
    try {
      const data = await listDocuments();
      setDocuments(data.files);
    } catch (err) {
      setError((err as Error).message);
    }
  }, []);

  useEffect(() => { fetchDocs(); }, [fetchDocs]);

  async function handleFiles(files: File[]) {
    setUploading(true);
    setError(null);
    try {
      await uploadFiles(files);
      await fetchDocs();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setUploading(false);
    }
  }

  async function handleDelete(filename: string) {
    setError(null);
    try {
      await deleteDocument(filename);
      await fetchDocs();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <div className="p-6">
      <h2 className="mb-6 text-lg font-semibold">Documents</h2>
      {error && (
        <p className="mb-4 rounded-lg bg-red-900/30 px-4 py-2 text-sm text-red-400">
          {error}
        </p>
      )}
      <DropZone onFiles={handleFiles} uploading={uploading} />
      <div className="mt-8">
        <DocumentTable documents={documents} onDelete={handleDelete} />
      </div>
    </div>
  );
}
```

**Step 4: Verify build**

```bash
pnpm --filter frontend build
# Expected: builds cleanly
```

**Step 5: Commit**

```bash
git add apps/frontend/app/documents/ apps/frontend/components/documents/
git commit -m "feat: implement document management page with drag-and-drop upload"
```

---

## Task 13: Implement models page

**Files:**
- Create: `apps/frontend/app/models/page.tsx`
- Create: `apps/frontend/components/models/ModelCard.tsx`

**Step 1: Create apps/frontend/components/models/ModelCard.tsx**

```tsx
"use client";

import type { ModelInfo } from "@rag/types";

export function ModelCard({
  model,
  onPull,
  onDelete,
  pulling,
  pullProgress,
}: {
  model: ModelInfo;
  onPull: () => void;
  onDelete: () => void;
  pulling: boolean;
  pullProgress: string;
}) {
  return (
    <div className="flex items-center justify-between rounded-xl border border-gray-800 bg-gray-900 px-5 py-4">
      <div>
        <p className="font-medium text-gray-100">{model.name}</p>
        <p className="text-xs text-gray-500">
          {model.downloaded ? `${model.size_mb} MB` : "Not downloaded"}
        </p>
        {pulling && (
          <p className="mt-1 text-xs text-brand-400">{pullProgress || "starting…"}</p>
        )}
      </div>
      <div className="flex gap-2">
        {!model.downloaded && !pulling && (
          <button
            type="button"
            onClick={onPull}
            className="rounded-lg bg-brand-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-brand-500 transition"
          >
            Download
          </button>
        )}
        {model.downloaded && (
          <button
            type="button"
            onClick={onDelete}
            className="rounded-lg border border-red-800 px-3 py-1.5 text-xs font-medium text-red-400 hover:bg-red-900/30 transition"
          >
            Delete
          </button>
        )}
      </div>
    </div>
  );
}
```

**Step 2: Create apps/frontend/app/models/page.tsx**

```tsx
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ModelCard } from "@/components/models/ModelCard";
import { deleteModel, getModelStatus, listModels, pullModel } from "@/lib/api";
import type { ModelInfo } from "@rag/types";

export default function ModelsPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [pulling, setPulling] = useState<Record<string, boolean>>({});
  const [pullProgress, setPullProgress] = useState<Record<string, string>>({});
  const pollRefs = useRef<Record<string, ReturnType<typeof setInterval>>>({});

  const fetchModels = useCallback(async () => {
    try {
      const data = await listModels();
      setModels(data.models);
    } catch (err) {
      setError((err as Error).message);
    }
  }, []);

  useEffect(() => {
    fetchModels();
    return () => {
      for (const t of Object.values(pollRefs.current)) clearInterval(t);
    };
  }, [fetchModels]);

  async function handlePull(name: string) {
    setError(null);
    setPulling((p) => ({ ...p, [name]: true }));
    try {
      await pullModel({ model: name });
      pollRefs.current[name] = setInterval(async () => {
        try {
          const status = await getModelStatus(name);
          setPullProgress((p) => ({ ...p, [name]: status.progress }));
          if (status.status === "completed") {
            clearInterval(pollRefs.current[name]);
            setPulling((p) => ({ ...p, [name]: false }));
            await fetchModels();
          } else if (status.status === "error") {
            clearInterval(pollRefs.current[name]);
            setPulling((p) => ({ ...p, [name]: false }));
            setError(`Pull failed: ${status.progress}`);
          }
        } catch {
          clearInterval(pollRefs.current[name]);
          setPulling((p) => ({ ...p, [name]: false }));
        }
      }, 2000);
    } catch (err) {
      setPulling((p) => ({ ...p, [name]: false }));
      setError((err as Error).message);
    }
  }

  async function handleDelete(name: string) {
    setError(null);
    try {
      await deleteModel(name);
      await fetchModels();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <div className="p-6">
      <h2 className="mb-6 text-lg font-semibold">Models</h2>
      {error && (
        <p className="mb-4 rounded-lg bg-red-900/30 px-4 py-2 text-sm text-red-400">
          {error}
        </p>
      )}
      <div className="space-y-3">
        {models.map((model) => (
          <ModelCard
            key={model.name}
            model={model}
            onPull={() => handlePull(model.name)}
            onDelete={() => handleDelete(model.name)}
            pulling={pulling[model.name] ?? false}
            pullProgress={pullProgress[model.name] ?? ""}
          />
        ))}
      </div>
    </div>
  );
}
```

**Step 3: Verify build**

```bash
pnpm --filter frontend build
# Expected: builds cleanly, no TypeScript errors
```

**Step 4: Commit**

```bash
git add apps/frontend/app/models/ apps/frontend/components/models/
git commit -m "feat: implement models page with pull/delete and progress polling"
```

---

## Task 14: Create frontend Dockerfile

**Files:**
- Create: `apps/frontend/Dockerfile`

The build context is the **repo root** (to include `packages/`). Configured in docker-compose in the next task.

**Step 1: Create apps/frontend/Dockerfile**

```dockerfile
# ── Builder ───────────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

RUN npm install -g pnpm

WORKDIR /app

# Copy workspace manifests first for layer caching.
COPY package.json pnpm-workspace.yaml pnpm-lock.yaml ./
COPY packages/ packages/
COPY apps/frontend/ apps/frontend/

RUN pnpm install --frozen-lockfile
RUN pnpm --filter frontend build

# ── Runtime ───────────────────────────────────────────────────────────────────
FROM node:20-alpine AS runtime

ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

WORKDIR /app

# Next.js standalone output includes its own node server.
COPY --from=builder /app/apps/frontend/.next/standalone ./
COPY --from=builder /app/apps/frontend/.next/static ./apps/frontend/.next/static
COPY --from=builder /app/apps/frontend/public ./apps/frontend/public

EXPOSE 3000

CMD ["node", "apps/frontend/server.js"]
```

**Step 2: Commit**

```bash
git add apps/frontend/Dockerfile
git commit -m "feat: add multi-stage Next.js frontend Dockerfile"
```

---

## Task 15: Update docker-compose.yml

**Files:**
- Modify: `docker-compose.yml`

**Step 1: Read current docker-compose.yml to understand existing structure**

The current file has `ollama` and `rag` services. We rename `rag` → `backend` and add `frontend`.

**Step 2: Replace docker-compose.yml**

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: rag-ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_NUM_THREADS=4
      - OLLAMA_KEEP_ALIVE=-1
      - OLLAMA_NOPRUNE=1
      - OLLAMA_FLASH_ATTENTION=1
    healthcheck:
      test: ["CMD", "ollama", "list"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 15s

  backend:
    build:
      context: apps/backend
      dockerfile: Dockerfile
    container_name: rag-backend
    restart: unless-stopped
    depends_on:
      ollama:
        condition: service_healthy
    ports:
      - "8443:8443"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - CHUNK_STRATEGY=semantic
      - CORS_ORIGINS=http://localhost:3000
    volumes:
      - ./apps/backend/documents:/app/documents
      - chroma_data:/app/chroma_db
      - ./certs:/app/certs
    healthcheck:
      test: ["CMD", "curl", "-k", "-f", "https://localhost:8443/api/v1/health"]
      interval: 15s
      timeout: 5s
      retries: 5
      start_period: 30s

  frontend:
    build:
      context: .
      dockerfile: apps/frontend/Dockerfile
    container_name: rag-frontend
    restart: unless-stopped
    depends_on:
      backend:
        condition: service_healthy
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=https://localhost:8443

volumes:
  ollama_data:
  chroma_data:
```

**Step 3: Create apps/frontend/.env.local for local dev**

```bash
cat > apps/frontend/.env.local << 'EOF'
NEXT_PUBLIC_API_URL=https://localhost:8443
EOF
```

Add `.env.local` to `.gitignore` in `apps/frontend/`:

Create `apps/frontend/.gitignore`:
```
.env.local
.next/
node_modules/
```

**Step 4: Test Docker build locally**

```bash
docker compose build
# Expected: all three images build without errors
```

**Step 5: Commit**

```bash
git add docker-compose.yml apps/frontend/.env.local apps/frontend/.gitignore
git commit -m "feat: update docker-compose with frontend service and updated backend context"
```

---

## Task 16: Update install.sh and uninstall.sh

**Files:**
- Modify: `install.sh`
- Modify: `uninstall.sh`

**Step 1: Find all path references in install.sh that need updating**

```bash
grep -n "documents\|docker-compose\|Dockerfile\|scripts/" install.sh | head -40
```

**Step 2: Update install.sh — change document and script path references**

Any reference to `./documents` or copying the `Dockerfile` from root should point to `apps/backend/`. Key patterns to update:

- `./documents` → `./apps/backend/documents` (volume path now matches docker-compose)
- References to `docker compose exec rag` → `docker compose exec backend`
- Any direct copy of `Dockerfile` from root → now at `apps/backend/Dockerfile`

Read the full file and apply the specific replacements needed. The exact edits depend on the current install.sh content, which references paths from repo root.

**Step 3: Run format and lint on the backend**

```bash
cd apps/backend && uv run ruff format src/ tests/ && uv run ruff check src/ tests/ --fix && uv run pytest
# Expected: all tests pass
```

**Step 4: Final smoke test — start services and hit health endpoint**

```bash
docker compose up -d ollama backend
# Wait ~30s for backend to become healthy
curl -k https://localhost:8443/api/v1/health
# Expected: {"status":"degraded",...} or "healthy" if Ollama is ready
```

**Step 5: Commit**

```bash
git add install.sh uninstall.sh
git commit -m "chore: update install/uninstall scripts for monorepo paths"
```

---

## Done

After Task 16, `turbo run dev` starts both the FastAPI backend and Next.js frontend.
`docker compose up` starts all three services.
The UI is at **http://localhost:3000** and the API at **https://localhost:8443**.
