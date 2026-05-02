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
