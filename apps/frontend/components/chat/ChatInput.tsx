"use client";

import { useRef } from "react";

export function ChatInput({
  onSubmit,
  disabled,
}: {
  onSubmit: (question: string) => void;
  disabled: boolean;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  function submit() {
    const value = ref.current?.value.trim();
    if (!value || disabled) return;
    onSubmit(value);
    if (ref.current) ref.current.value = "";
  }

  return (
    <div className="border-t border-gray-800 p-4">
      <div className="flex items-end gap-3 rounded-xl border border-gray-700 bg-gray-800 px-4 py-3">
        <textarea
          ref={ref}
          rows={1}
          placeholder="Ask a question about your documents…"
          disabled={disabled}
          onKeyDown={handleKeyDown}
          className="flex-1 resize-none bg-transparent text-sm text-gray-100 placeholder-gray-500 outline-none disabled:opacity-50"
        />
        <button
          type="button"
          onClick={submit}
          disabled={disabled}
          className="rounded-lg bg-brand-500 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-brand-600 disabled:opacity-40"
        >
          Send
        </button>
      </div>
      <p className="mt-2 text-center text-xs text-gray-600">
        Enter to send · Shift+Enter for new line
      </p>
    </div>
  );
}
