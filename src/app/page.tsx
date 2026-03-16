"use client";

import {
  transformersJS,
  TransformersJSEmbeddingModel,
} from "@browser-ai/transformers-js";
import { embed } from "ai";
import { useEffect, useMemo, useState } from "react";

export default function Home() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState<number[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState<number | null>(null);

  const embeddingModel = useMemo(() => {
    return transformersJS.embedding("Xenova/all-MiniLM-L6-v2", {
      device: "wasm",
      dtype: "q4",
    });
  }, []);

  useEffect(() => {
    (embeddingModel as TransformersJSEmbeddingModel)
      .createSessionWithProgress((progress: number) => {
        setDownloadProgress(progress);
      })
      .then(() => {
        setModelReady(true);
        setDownloadProgress(null);
      });
  }, [embeddingModel]);

  async function handleEmbed() {
    if (!embeddingModel || !input.trim()) return;
    setLoading(true);
    try {
      const { embedding } = await embed({
        model: embeddingModel,
        value: input,
      });
      setResult(embedding);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 font-sans dark:bg-black">
      <main className="flex min-h-screen w-full max-w-3xl flex-col py-8 px-4 bg-white dark:bg-zinc-900">
        <div className="flex items-center gap-2 mb-4">
          <h1 className="text-xl font-semibold">Embedding playground</h1>
          {modelReady ? (
            <span className="text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300">
              Model ready
            </span>
          ) : (
            <span className="text-xs px-2 py-0.5 rounded-full bg-zinc-200 text-zinc-500 dark:bg-zinc-700 dark:text-zinc-400">
              {downloadProgress !== null
                ? `Loading… ${Math.round(downloadProgress * 100)} %`
                : "Initializing…"}
            </span>
          )}
        </div>

        {downloadProgress !== null && !modelReady && (
          <div className="mb-4 h-1.5 w-full rounded-full bg-zinc-200 dark:bg-zinc-700 overflow-hidden">
            <div
              className="h-full rounded-full bg-blue-500 transition-all duration-200"
              style={{ width: `${Math.round(downloadProgress * 100)}%` }}
            />
          </div>
        )}

        <div className="flex gap-2 mb-4">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Text to embed..."
            className="flex-1 rounded-lg border px-4 py-2 dark:bg-zinc-800 dark:border-zinc-700"
          />
          <button
            onClick={handleEmbed}
            disabled={loading || !input.trim() || !modelReady}
            className="rounded-lg bg-blue-600 px-4 py-2 text-white disabled:opacity-50"
          >
            {loading ? "Computing..." : "Embed"}
          </button>
        </div>

        {result && (
          <div className="rounded-lg bg-zinc-100 dark:bg-zinc-800 p-4 text-xs font-mono overflow-auto">
            <p className="mb-1 text-zinc-500">Dimensions: {result.length}</p>
            <p className="break-all">
              [
              {result
                .slice(0, 16)
                .map((v) => v.toFixed(4))
                .join(", ")}
              , …]
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
