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
