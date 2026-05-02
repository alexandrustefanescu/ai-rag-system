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
