"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { signOut, useSession } from "next-auth/react";
import { useEffect, useState } from "react";
import type { Conversation } from "@/lib/api";
import { listConversations } from "@/lib/api";

export const dynamic = "force-dynamic";

const links = [
  { href: "/chat", label: "Chat", icon: "💬" },
  { href: "/documents", label: "Documents", icon: "📄" },
  { href: "/models", label: "Models", icon: "🤖" },
];

export function Sidebar() {
    const pathname = usePathname();
    const { data: session, status } = useSession();

    const [conversations, setConversations] = useState<Conversation[]>([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (status !== "authenticated" || !session?.user) return;
        setLoading(true);
        listConversations(session.user.id)
            .then((data) => setConversations(data.conversations))
            .catch(() => setConversations([]))
            .finally(() => setLoading(false));
    }, [session]);

    return (
        <nav className="flex w-52 flex-col border-r border-gray-800 bg-gray-900 p-4">
            <div className="mb-4">
                <h1 className="text-sm font-semibold tracking-widest text-gray-400 uppercase">
                    RAG System
                </h1>
            </div>
            <ul className="space-y-1">
                {links.map(({ href, label, icon }) => (
                    <li key={href}>
                        <Link
                            href={href}
                            className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors ${
                                pathname.startsWith(href)
                                    ? "bg-brand-500/20 text-brand-300"
                                    : "text-gray-400 hover:bg-gray-800 hover:text-gray-100"
                            }`}
                        >
                            <span>{icon}</span>
                            {label}
                        </Link>
                    </li>
                ))}
            </ul>
            {session?.user && (
                <div className="mt-4 flex-1 overflow-y-auto">
                    <h2 className="mb-2 text-xs font-semibold uppercase text-gray-500">
                        Conversations
                    </h2>
                    {loading ? (
                        <p className="text-xs text-gray-500">Loading...</p>
                    ) : (
                        <ul className="space-y-1">
                            {conversations.map((conv) => (
                                <li key={conv.id}>
                                    <Link
                                        href={`/chat?conv=${conv.id}`}
                                        className={`block truncate rounded px-2 py-1 text-xs ${
                                            pathname === "/chat" &&
                                            new URLSearchParams(window.location.search).get("conv") === conv.id
                                                ? "bg-brand-500/20 text-brand-300"
                                                : "text-gray-400 hover:bg-gray-800"
                                        }`}
                                    >
                                        {conv.title}
                                    </Link>
                                </li>
                            ))}
                        </ul>
                    )}
                </div>
            )}
            {session?.user && (
                <div className="mt-auto border-t border-gray-800 pt-4">
                    <p className="truncate text-xs text-gray-400">
                        {session.user.email}
                    </p>
                    <button
                        onClick={() => signOut()}
                        className="mt-2 w-full rounded bg-gray-800 px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-700"
                    >
                        Sign out
                    </button>
                </div>
            )}
        </nav>
    );
}
