"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/chat", label: "Chat", icon: "💬" },
  { href: "/documents", label: "Documents", icon: "📄" },
  { href: "/models", label: "Models", icon: "🤖" },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <nav className="flex w-52 flex-col border-r border-gray-800 bg-gray-900 p-4">
      <div className="mb-8">
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
    </nav>
  );
}
