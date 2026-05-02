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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
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

export async function deleteDocument(
  filename: string,
): Promise<DeleteResponse> {
  return request<DeleteResponse>(
    `/documents/${encodeURIComponent(filename)}`,
    { method: "DELETE" },
  );
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
