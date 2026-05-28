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

function getHeaders(token?: string): Record<string, string> {
    const headers: Record<string, string> = {
        "Content-Type": "application/json",
    };
    if (token) {
        headers["Authorization"] = `Bearer ${token}`;
    }
    return headers;
}

async function request<T>(
    path: string,
    init?: RequestInit,
    token?: string,
): Promise<T> {
    const res = await fetch(`${API_URL}/api/v1${path}`, {
        ...init,
        headers: { ...getHeaders(token), ...init?.headers },
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

export async function ask(
    body: AskRequest,
    token?: string,
): Promise<AskResponse> {
    return request<AskResponse>(
        "/ask",
        { method: "POST", body: JSON.stringify(body) },
        token,
    );
}

export async function* streamAsk(
    body: AskRequest,
    signal?: AbortSignal,
    token?: string,
): AsyncGenerator<string> {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) {
        headers["Authorization"] = `Bearer ${token}`;
    }
    const res = await fetch(`${API_URL}/api/v1/ask/stream`, {
        method: "POST",
        headers,
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

export async function listDocuments(
    token?: string,
): Promise<DocumentListResponse> {
    return request<DocumentListResponse>("/documents", undefined, token);
}

export async function uploadFiles(
    files: File[],
    token?: string,
): Promise<UploadResponse> {
    const form = new FormData();
    for (const file of files) form.append("files", file);
    const headers: Record<string, string> = {};
    if (token) {
        headers["Authorization"] = `Bearer ${token}`;
    }
    const res = await fetch(`${API_URL}/api/v1/upload`, {
        method: "POST",
        headers,
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
    token?: string,
): Promise<DeleteResponse> {
    return request<DeleteResponse>(
        `/documents/${encodeURIComponent(filename)}`,
        { method: "DELETE" },
        token,
    );
}

export async function ingest(token?: string): Promise<IngestResponse> {
    return request<IngestResponse>("/ingest", { method: "POST" }, token);
}

export async function listModels(): Promise<ModelListResponse> {
    return request<ModelListResponse>("/models");
}

export async function pullModel(
    body: PullRequest,
): Promise<PullResponse> {
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

// --- Chat History API ---

export interface Conversation {
    id: string;
    title: string;
    created_at: string;
    updated_at: string;
}

export interface Message {
    id: string;
    role: string;
    content: string;
    sources?: string | null;
    created_at: string;
}

export async function listConversations(
    token?: string,
): Promise<{ conversations: Conversation[] }> {
    return request<{ conversations: Conversation[] }>(
        "/conversations",
        undefined,
        token,
    );
}

export async function createConversation(
    title?: string,
    token?: string,
): Promise<Conversation> {
    return request<Conversation>(
        "/conversations",
        {
            method: "POST",
            body: JSON.stringify({ title }),
        },
        token,
    );
}

export async function getMessages(
    convId: string,
    token?: string,
): Promise<Message[]> {
    return request<Message[]>(
        `/conversations/${convId}/messages`,
        undefined,
        token,
    );
}
