"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { ChatInput } from "@/components/chat/ChatInput";
import { MessageList, type Message } from "@/components/chat/MessageList";
import { streamAsk, ask, createConversation, getMessages, type Conversation } from "@/lib/api";

export const dynamic = "force-dynamic";

export default function ChatPage() {
    const { data: session, status } = useSession();
    const router = useRouter();
    const [messages, setMessages] = useState<Message[]>([]);
    const [streaming, setStreaming] = useState(false);
    const [currentConv, setCurrentConv] = useState<Conversation | null>(null);
    const abortRef = useRef<AbortController | null>(null);

    const convId = new URLSearchParams(
        typeof window !== "undefined" ? window.location.search : "",
    ).get("conv");

    useEffect(() => {
        if (status === "unauthenticated") {
            router.push("/login");
        }
    }, [status, router]);

    useEffect(() => {
        if (!session?.user) return;
        if (!convId) {
            setMessages([]);
            setCurrentConv(null);
            return;
        }
        getMessages(convId, (session.user as { id: string }).id)
            .then((msgs) => {
                setMessages(
                    msgs.map((m) => ({
                        id: m.id,
                        role: m.role as "user" | "assistant",
                        content: m.content,
                    })),
                );
                setCurrentConv({ id: convId, title: "", created_at: "", updated_at: "" });
            })
            .catch(() => {
                setMessages([]);
            });
    }, [convId, session]);

    const handleSubmit = useCallback(
        async (question: string) => {
            if (!session?.user) return;

            const token = (session.user as { id: string }).id;
            const userMsg: Message = {
                id: crypto.randomUUID(),
                role: "user",
                content: question,
            };
            const assistantId = crypto.randomUUID();
            const assistantMsg: Message = {
                id: assistantId,
                role: "assistant",
                content: "",
                streaming: true,
            };

            setMessages((prev) => [...prev, userMsg, assistantMsg]);
            setStreaming(true);

            const controller = new AbortController();
            abortRef.current = controller;

            try {
                let fullContent = "";
                for await (const chunk of streamAsk(
                    { question, conversation_id: convId ?? undefined },
                    controller.signal,
                    token,
                )) {
                    fullContent += chunk;
                    setMessages((prev) =>
                        prev.map((m) =>
                            m.id === assistantId ? { ...m, content: fullContent } : m,
                        ),
                    );
                }
            } catch (err) {
                if ((err as Error).name !== "AbortError") {
                    setMessages((prev) =>
                        prev.map((m) =>
                            m.id === assistantId
                                ? {
                                      ...m,
                                      content: `Error: ${(err as Error).message}`,
                                      streaming: false,
                                  }
                                : m,
                        ),
                    );
                }
            } finally {
                setMessages((prev) =>
                    prev.map((m) =>
                        m.id === assistantId ? { ...m, streaming: false } : m,
                    ),
                );
                setStreaming(false);
            }
        },
        [session, convId],
    );

    if (status === "loading") {
        return (
            <div className="flex h-full items-center justify-center">
                Loading...
            </div>
        );
    }

    return (
        <div className="flex h-full flex-col">
            <header className="border-b border-gray-800 px-6 py-4">
                <h2 className="text-lg font-semibold">Chat</h2>
            </header>
            <div className="flex-1 overflow-y-auto">
                {messages.length === 0 ? (
                    <div className="flex h-full items-center justify-center text-gray-500 text-sm">
                        Upload documents, then ask a question.
                    </div>
                ) : (
                    <MessageList messages={messages} />
                )}
            </div>
            <ChatInput onSubmit={handleSubmit} disabled={streaming} />
        </div>
    );
}
