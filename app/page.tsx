"use client";

import { useCallback, useMemo, useState } from "react";

export default function Home() {
  const [droppedImages, setDroppedImages] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [aiText, setAiText] = useState<string | null>(null);
  const [inferStatus, setInferStatus] = useState<"idle" | "loading" | "error">("idle");
  const [inferError, setInferError] = useState<string | null>(null);

  const imageListText = useMemo(() => {
    if (droppedImages.length === 0) {
      return "Drop images here to attach them.";
    }

    return droppedImages.map((file) => file.name).join(", ");
  }, [droppedImages]);

  const runInference = useCallback(async (files: File[]) => {
    if (files.length === 0) return;
    const file = files[0];
    setInferStatus("loading");
    setInferError(null);
    setAiText(null);
    try {
      const body = new FormData();
      body.append("image", file);
      const res = await fetch("/api/infer", { method: "POST", body });
      const data = (await res.json()) as {
        ok?: boolean;
        summary?: string;
        error?: string;
        stderr?: string;
        payload?: unknown;
      };
      if (!res.ok || !data.ok) {
        setInferStatus("error");
        setInferError(data.error ?? `Request failed (${res.status})`);
        return;
      }
      const summary = data.summary?.trim() ?? "";
      const pretty =
        data.payload !== undefined && data.payload !== null
          ? JSON.stringify(data.payload, null, 2)
          : "";
      setAiText([summary, pretty ? `\n--- JSON ---\n${pretty}` : ""].filter(Boolean).join("\n"));
      setInferStatus("idle");
    } catch (e) {
      setInferStatus("error");
      setInferError(e instanceof Error ? e.message : "Inference request failed");
    }
  }, []);

  const handleDragOver: React.DragEventHandler<HTMLLabelElement> = (event) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave: React.DragEventHandler<HTMLLabelElement> = () => {
    setIsDragging(false);
  };

  const handleDrop: React.DragEventHandler<HTMLLabelElement> = (event) => {
    event.preventDefault();
    setIsDragging(false);
    const files = Array.from(event.dataTransfer.files).filter((file) =>
      file.type.startsWith("image/"),
    );
    if (files.length > 0) {
      setDroppedImages(files);
      void runInference(files);
    }
  };

  const handleFileSelect: React.ChangeEventHandler<HTMLInputElement> = (
    event,
  ) => {
    const files = Array.from(event.target.files ?? []).filter((file) =>
      file.type.startsWith("image/"),
    );
    if (files.length > 0) {
      setDroppedImages(files);
      void runInference(files);
    }
  };

  return (
    <div className="h-full w-full flex justify-end bg-transparent">
      <aside
        className="w-[320px] h-full rounded-l-2xl rounded-r-none overflow-hidden flex flex-col border-2 border-sky-300/90 shadow-2xl shadow-sky-300/25"
        style={{
          background: "rgba(125, 211, 252, 0.2)",
          backdropFilter: "blur(18px)",
          WebkitBackdropFilter: "blur(18px)",
        }}
      >
        <div className="px-4 py-3 border-b border-white/[0.06]">
          <h1 className="text-[11px] font-bold tracking-wide text-blue-400">
            AI RESPONSE PANEL
          </h1>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          <div className="h-full min-h-[120px] rounded-xl border border-white/[0.06] bg-black/20 p-3">
            <p className="text-[10px] font-semibold uppercase tracking-wide text-zinc-500 mb-2">
              AI response
            </p>
            {inferStatus === "loading" && (
              <p className="text-[12px] leading-relaxed text-sky-300/90 font-mono animate-pulse">
                Running quad YOLO inference…
              </p>
            )}
            {inferStatus === "error" && inferError && (
              <p className="text-[12px] leading-relaxed text-red-300/90 font-mono whitespace-pre-wrap">
                {inferError}
              </p>
            )}
            {inferStatus !== "loading" && aiText && (
              <pre className="text-[11px] leading-relaxed text-zinc-200 font-mono whitespace-pre-wrap break-words">
                {aiText}
              </pre>
            )}
            {inferStatus !== "loading" && !aiText && !inferError && (
              <p className="text-[12px] leading-relaxed text-zinc-500 font-mono">
                Drop an image below to run the Marvel Rivals detection suite (hero,
                HP, UI, friend/foe) and show results here.
              </p>
            )}
          </div>
        </div>

        <div className="p-3 border-t border-white/[0.06]">
          <label
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`block rounded-lg border px-3 py-2 text-center cursor-pointer transition-colors ${
              isDragging
                ? "border-blue-400 bg-blue-500/10"
                : "border-white/[0.10] bg-black/20 hover:border-blue-300/70"
            }`}
          >
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={handleFileSelect}
              className="hidden"
            />
            <p className="text-[11px] font-semibold tracking-wide text-zinc-200">
              Drag &amp; Drop Images
            </p>
            <p className="mt-1 text-[10px] text-zinc-400 break-words">
              {imageListText}
            </p>
          </label>
        </div>
      </aside>
    </div>
  );
}
