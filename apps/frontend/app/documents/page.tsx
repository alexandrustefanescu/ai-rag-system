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

  useEffect(() => {
    fetchDocs();
  }, [fetchDocs]);

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
