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
          <p className="mt-1 text-xs text-brand-400">
            {pullProgress || "starting…"}
          </p>
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
