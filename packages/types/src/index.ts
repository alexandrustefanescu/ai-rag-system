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
