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
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
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
      <p className="text-xs text-gray-600">
        Supported: .txt .md .pdf (max 50 MB each)
      </p>
    </div>
  );
}
