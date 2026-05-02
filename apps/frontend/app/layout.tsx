import type { Metadata } from "next";
import { Sidebar } from "@/components/nav/Sidebar";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI RAG System",
  description: "Local Retrieval-Augmented Generation powered by Ollama",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="flex h-screen bg-gray-950 text-gray-100 antialiased">
        <Sidebar />
        <main className="flex-1 overflow-auto">{children}</main>
      </body>
    </html>
  );
}
