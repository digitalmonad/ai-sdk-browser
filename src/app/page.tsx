"use client";

import {
  transformersJS,
  TransformersJSEmbeddingModel,
} from "@browser-ai/transformers-js";
import { embed } from "ai";
import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

const MODEL_ID = "Xenova/all-MiniLM-L6-v2";

export default function Home() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState<number[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState<number | null>(null);
  const [isCached, setIsCached] = useState<boolean | null>(null);
  const [inferenceTime, setInferenceTime] = useState<number | null>(null);

  const embeddingModel = useMemo(() => {
    return transformersJS.embedding(MODEL_ID, {
      device: "wasm",
      dtype: "q4",
    });
  }, []);

  const checkCache = useCallback(async () => {
    if (!("caches" in window)) {
      setIsCached(false);
      return;
    }
    try {
      const cacheNames = await caches.keys();
      for (const name of cacheNames) {
        const cache = await caches.open(name);
        const keys = await cache.keys();
        if (keys.some((req) => req.url.includes(MODEL_ID.split("/")[1]))) {
          setIsCached(true);
          return;
        }
      }
      setIsCached(false);
    } catch {
      setIsCached(false);
    }
  }, []);

  useEffect(() => {
    checkCache();
  }, [checkCache]);

  async function handleDownloadModel() {
    setDownloading(true);
    toast.loading("Downloading model…", { id: "model-download" });
    try {
      await (
        embeddingModel as TransformersJSEmbeddingModel
      ).createSessionWithProgress((progress: number) => {
        setDownloadProgress(progress);
        toast.loading(`Downloading model… ${Math.round(progress * 100)} %`, {
          id: "model-download",
        });
      });
      setModelReady(true);
      setDownloadProgress(null);
      setIsCached(true);
      toast.success("Model ready!", { id: "model-download" });
    } catch {
      toast.error("Failed to load model", { id: "model-download" });
    } finally {
      setDownloading(false);
    }
  }

  async function handleClearCache() {
    if (!("caches" in window)) {
      toast.error("Cache API is not supported in this browser");
      return;
    }
    try {
      const cacheNames = await caches.keys();
      let deleted = 0;
      for (const name of cacheNames) {
        const cache = await caches.open(name);
        const keys = await cache.keys();
        const modelKeys = keys.filter((req) =>
          req.url.includes(MODEL_ID.split("/")[1]),
        );
        await Promise.all(modelKeys.map((req) => cache.delete(req)));
        deleted += modelKeys.length;
      }
      setIsCached(false);
      setModelReady(false);
      toast.success(deleted > 0 ? "Cache cleared" : "Nothing to clear");
    } catch {
      toast.error("Failed to clear cache");
    }
  }

  async function handleEmbed() {
    if (!embeddingModel || !input.trim()) return;
    setLoading(true);
    setInferenceTime(null);
    try {
      const start = performance.now();
      const { embedding } = await embed({
        model: embeddingModel,
        value: input,
      });
      const end = performance.now();
      const ms = end - start;
      setResult(embedding);
      setInferenceTime(ms);
      toast.success(`Embedding computed (${Math.round(ms)} ms)`);
    } catch {
      toast.error("Failed to compute embedding");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 font-sans dark:bg-zinc-900">
      <main className="flex min-h-screen w-full max-w-3xl flex-col py-8 px-4">
        <div className="flex items-center gap-2 mb-6">
          <h1 className="text-xl font-semibold">Embedding playground</h1>
          {modelReady ? (
            <span className="text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300">
              Model ready
            </span>
          ) : downloading ? (
            <span className="text-xs px-2 py-0.5 rounded-full bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300">
              {downloadProgress !== null ? `Downloading` : "Starting"}
            </span>
          ) : (
            <span className="text-xs px-2 py-0.5 rounded-full bg-zinc-200 text-zinc-500 dark:bg-zinc-700 dark:text-zinc-400">
              Not loaded
            </span>
          )}
        </div>

        {/* Model info card */}
        <div className="mb-6 rounded-lg border border-zinc-200 dark:border-zinc-700 p-4 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
              Model info
            </span>
            {isCached && !modelReady && (
              <button
                onClick={handleClearCache}
                className="text-xs p-2 rounded-sm bg-red-600/30 border border-red-600 hover:bg-red-600/50 transition-colors cursor-pointer"
              >
                Clear cache
              </button>
            )}
            {modelReady && (
              <button
                onClick={handleClearCache}
                className="text-xs p-2 rounded-sm bg-red-600/30 border border-red-600 hover:bg-red-600/50 transition-colors cursor-pointer"
              >
                Clear cache &amp; unload
              </button>
            )}
          </div>

          <div className="grid grid-cols-[6rem_1fr] gap-x-4 gap-y-1.5 text-xs">
            <span className="text-zinc-500 dark:text-zinc-400">Model ID</span>
            <span className="font-mono text-zinc-700 dark:text-zinc-300">
              {MODEL_ID}
            </span>
            <span className="text-zinc-500 dark:text-zinc-400">Storage</span>
            <span className="font-mono text-zinc-700 dark:text-zinc-300">
              Cache Storage API
            </span>
            <span className="text-zinc-500 dark:text-zinc-400">Source</span>
            <a
              href={`https://huggingface.co/${MODEL_ID}`}
              target="_blank"
              rel="noopener noreferrer"
              className="font-mono text-zinc-700 dark:text-zinc-300 break-all underline hover:opacity-80"
            >
              huggingface.co/{MODEL_ID}
            </a>
            <span className="text-zinc-500 dark:text-zinc-400">Cached</span>
            <span>
              {isCached === null ? (
                <span className="text-zinc-400">Checking…</span>
              ) : isCached ? (
                <span className="text-green-600 dark:text-green-400 font-medium">
                  Yes — served from browser cache
                </span>
              ) : (
                <span className="text-zinc-400">No</span>
              )}
            </span>
          </div>

          {!modelReady && (
            <>
              {downloading && downloadProgress !== null && (
                <div className="h-1.5 w-full rounded-full bg-zinc-200 dark:bg-zinc-700 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-blue-500 transition-all duration-200"
                    style={{ width: `${Math.round(downloadProgress * 100)}%` }}
                  />
                </div>
              )}
              <button
                onClick={handleDownloadModel}
                disabled={downloading}
                className="w-full rounded-lg bg-blue-600 px-4 py-2 text-sm text-white disabled:opacity-50 hover:bg-blue-700 transition-colors cursor-pointer"
              >
                {downloading
                  ? downloadProgress !== null
                    ? `Downloading…`
                    : "Starting…"
                  : isCached
                    ? "Load model from cache"
                    : "Download model"}
              </button>
            </>
          )}
        </div>

        <div className="flex gap-2 mb-4">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                handleEmbed();
              }
            }}
            placeholder="Text to embed..."
            className="flex-1 rounded-sm border p-2 dark:bg-zinc-800 dark:border-zinc-700"
          />
          <button
            onClick={handleEmbed}
            disabled={loading || !input.trim() || !modelReady}
            className="rounded-sm bg-blue-600/30 border border-blue-600 hover:bg-blue-600/50 disabled:hover:bg-blue-600/30 px-4 py-2 text-white disabled:opacity-50 cursor-pointer disabled:cursor-default"
          >
            {loading ? "Computing..." : "Embed"}
          </button>
        </div>

        <div className="mb-4 text-xs text-zinc-500">
          {loading ? (
            <span>Running inference…</span>
          ) : inferenceTime !== null ? (
            <span>
              Inference duration:{" "}
              <span className="font-mono text-zinc-700 dark:text-zinc-300">
                {Math.round(inferenceTime)} ms
              </span>
            </span>
          ) : (
            <span>Last inference: —</span>
          )}
        </div>

        {result && (
          <div className="rounded-lg bg-zinc-100 dark:bg-zinc-800 p-4 text-xs font-mono overflow-auto">
            <p className="mb-1 text-zinc-500">Dimensions: {result.length}</p>
            <p className="break-all">
              [{result.map((v) => v.toFixed(4)).join(", ")}]
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
