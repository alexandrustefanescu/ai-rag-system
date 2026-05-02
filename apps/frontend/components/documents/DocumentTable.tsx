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
