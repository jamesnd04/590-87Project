"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

function base64PngToFile(base64: string, filename: string): File {
  const bin = atob(base64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) {
    bytes[i] = bin.charCodeAt(i);
  }
  return new File([bytes], filename, { type: "image/png" });
}

export default function Home() {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioObjectUrlRef = useRef<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [userInput, setUserInput] = useState("");
  const [agentText, setAgentText] = useState<string | null>(null);
  const [toolTrace, setToolTrace] = useState<string | null>(null);
  const [agentStatus, setAgentStatus] = useState<"idle" | "loading" | "error">("idle");
  const [agentError, setAgentError] = useState<string | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [layoutPreset, setLayoutPreset] = useState<string>("");
  const [hasElectronIpc, setHasElectronIpc] = useState(false);
  const [teamContext, setTeamContext] = useState<{
    yourTeam: string[];
    enemyTeam: string[];
  } | null>(null);

  const imageLabelText = useMemo(() => {
    if (!selectedImage) {
      return "Drop an image here to attach it.";
    }
    return selectedImage.name;
  }, [selectedImage]);

  const submitAgent = useCallback(async (imageOverride?: File | null) => {
    const imageToUse = imageOverride ?? selectedImage;
    if (!userInput.trim() && !imageToUse) {
      setAgentStatus("error");
      setAgentError("Enter text and/or attach an image.");
      return;
    }
    setAgentStatus("loading");
    setAgentError(null);
    setAgentText(null);
    setToolTrace(null);
    try {
      const body = new FormData();
      body.append("input", userInput);
      if (imageToUse) {
        body.append("image", imageToUse);
      }
      if (layoutPreset.trim()) {
        body.append("layout", layoutPreset.trim());
      }
      if (teamContext) {
        body.append("team_context", JSON.stringify(teamContext));
      }
      const res = await fetch("/api/agent", { method: "POST", body });
      const data = (await res.json()) as {
        ok?: boolean;
        error?: string;
        response?: string;
        intent?: string;
        tools?: Array<{ tool: string; ok: boolean; details?: string }>;
        imageContext?: { yourTeam?: string[]; enemyTeam?: string[] };
      };
      if (!res.ok || !data.ok) {
        setAgentStatus("error");
        setAgentError(data.error ?? `Request failed (${res.status})`);
        return;
      }
      setAgentText(data.response ?? "(No response)");
      if (data.imageContext) {
        setTeamContext({
          yourTeam: Array.isArray(data.imageContext.yourTeam) ? data.imageContext.yourTeam : [],
          enemyTeam: Array.isArray(data.imageContext.enemyTeam) ? data.imageContext.enemyTeam : [],
        });
      }
      const trace = [
        `Intent: ${data.intent ?? "unknown"}`,
        ...(Array.isArray(data.tools)
          ? data.tools.map((t) => `- ${t.tool}: ${t.ok ? "ok" : "failed"}${t.details ? ` (${t.details})` : ""}`)
          : []),
      ].join("\n");
      setToolTrace(trace);
      setAgentStatus("idle");
    } catch (e) {
      setAgentStatus("error");
      setAgentError(e instanceof Error ? e.message : "Agent request failed");
    }
  }, [layoutPreset, selectedImage, teamContext, userInput]);

  useEffect(() => {
    setHasElectronIpc(Boolean(window.ipc));
  }, []);

  useEffect(() => {
    const off = window.ipc?.onUnderlayScreenshot?.((base64) => {
      const name = `underlay-${Date.now()}.png`;
      const file = base64PngToFile(base64, name);
      setSelectedImage(file);
      void submitAgent(file);
    });
    return () => {
      off?.();
    };
  }, [submitAgent]);

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
      setSelectedImage(files[0]);
    }
  };

  const handleFileSelect: React.ChangeEventHandler<HTMLInputElement> = (
    event,
  ) => {
    const files = Array.from(event.target.files ?? []).filter((file) =>
      file.type.startsWith("image/"),
    );
    if (files.length > 0) {
      setSelectedImage(files[0]);
    }
  };

  const submitFromForm: React.FormEventHandler<HTMLFormElement> = (event) => {
    event.preventDefault();
    void submitAgent();
  };

  const handleSpeak = useCallback(() => {
    if (!agentText) {
      return;
    }
    if (isSpeaking) {
      audioRef.current?.pause();
      audioRef.current = null;
      if (audioObjectUrlRef.current) {
        URL.revokeObjectURL(audioObjectUrlRef.current);
        audioObjectUrlRef.current = null;
      }
      setIsSpeaking(false);
      return;
    }
    void (async () => {
      try {
        const body = new FormData();
        body.append("text", agentText);
        const response = await fetch("/api/speak", { method: "POST", body });
        if (!response.ok) {
          const err = (await response.json()) as { error?: string };
          throw new Error(err.error || "TTS request failed.");
        }
        const blob = await response.blob();
        const objectUrl = URL.createObjectURL(blob);
        audioObjectUrlRef.current = objectUrl;
        const audio = new Audio(objectUrl);
        audioRef.current = audio;
        audio.onended = () => {
          setIsSpeaking(false);
          if (audioObjectUrlRef.current) {
            URL.revokeObjectURL(audioObjectUrlRef.current);
            audioObjectUrlRef.current = null;
          }
          audioRef.current = null;
        };
        setIsSpeaking(true);
        await audio.play();
      } catch (error) {
        setIsSpeaking(false);
        setAgentStatus("error");
        setAgentError(error instanceof Error ? error.message : "OpenAI TTS playback failed.");
      }
    })();
  }, [agentText, isSpeaking]);

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

        <form className="flex-1 overflow-y-auto p-4 space-y-3" onSubmit={submitFromForm}>
          <div className="rounded-xl border border-white/[0.06] bg-black/20 p-3">
            <p className="text-[10px] font-semibold uppercase tracking-wide text-zinc-500 mb-2">
              Prompt
            </p>
            <textarea
              value={userInput}
              onChange={(event) => setUserInput(event.target.value)}
              placeholder="Ask a strategy question, request advice, or send text + image together."
              className="w-full min-h-[90px] resize-y rounded-md border border-white/[0.10] bg-black/30 px-2 py-2 text-[11px] text-zinc-200 font-mono focus:outline-none focus:ring-1 focus:ring-sky-400/60"
            />
            <div className="mt-2 flex items-center gap-2">
              <button
                type="button"
                onClick={handleSpeak}
                disabled={!agentText}
                className="rounded-md border border-white/[0.12] bg-black/40 px-2 py-1 text-[11px] text-zinc-200 disabled:opacity-50"
                title="Read response aloud using OpenAI TTS"
              >
                {isSpeaking ? "🔈 Stop" : "🔊 Read aloud"}
              </button>
            </div>
            {teamContext && (
              <div className="mt-2 rounded-md border border-blue-300/30 bg-blue-500/10 p-2">
                <p className="text-[10px] font-semibold uppercase tracking-wide text-blue-200">
                  Image + Prompt Context Linked
                </p>
                <p className="text-[10px] text-zinc-300 mt-1">
                  Your team: {teamContext.yourTeam.join(", ") || "(unknown)"}{" "}
                  | Enemy team: {teamContext.enemyTeam.join(", ") || "(unknown)"}
                </p>
              </div>
            )}
            <button
              type="submit"
              disabled={agentStatus === "loading"}
              className="mt-2 w-full rounded-md border border-blue-300/60 bg-blue-500/15 px-3 py-1.5 text-[11px] font-semibold tracking-wide text-blue-200 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {agentStatus === "loading" ? "Agent Running..." : "Run Agent"}
            </button>
          </div>

          <div className="h-full min-h-[120px] rounded-xl border border-white/[0.06] bg-black/20 p-3">
            <p className="text-[10px] font-semibold uppercase tracking-wide text-zinc-500 mb-2">
              AI response
            </p>
            {agentStatus === "loading" && (
              <p className="text-[12px] leading-relaxed text-sky-300/90 font-mono animate-pulse">
                Agent is reasoning with tool calls...
              </p>
            )}
            {agentStatus === "error" && agentError && (
              <p className="text-[12px] leading-relaxed text-red-300/90 font-mono whitespace-pre-wrap">
                {agentError}
              </p>
            )}
            {agentStatus !== "loading" && agentText && (
              <pre className="text-[11px] leading-relaxed text-zinc-200 font-mono whitespace-pre-wrap break-words">
                {agentText}
              </pre>
            )}
            {agentStatus !== "loading" && toolTrace && (
              <pre className="mt-3 text-[10px] leading-relaxed text-zinc-400 font-mono whitespace-pre-wrap break-words border-t border-white/[0.06] pt-3">
                {toolTrace}
              </pre>
            )}
            {agentStatus !== "loading" && !agentText && !agentError && (
              <p className="text-[12px] leading-relaxed text-zinc-500 font-mono">
                Ask a question, attach a screenshot, or do both. The agent decides when to run
                the YOLO tool and when to search your strategy guide markdown.
              </p>
            )}
          </div>
        </form>

        <div className="p-3 border-t border-white/[0.06] space-y-2">
          <div className="px-1">
            <label className="block text-[10px] font-semibold uppercase tracking-wide text-zinc-500 mb-1">
              Layout preset
            </label>
            <select
              value={layoutPreset}
              onChange={(e) => setLayoutPreset(e.target.value)}
              className="w-full rounded-md border border-white/[0.10] bg-black/40 px-2 py-1.5 text-[11px] text-zinc-200 font-mono focus:outline-none focus:ring-1 focus:ring-sky-400/60"
            >
              <option value="">Default (post-match 12-row)</option>
              <option value="config/layouts/post_match_1024.json">Post-match (explicit)</option>
              <option value="config/layouts/team_panel_1024.json">YOUR TEAM (6 rows)</option>
              <option value="config/layouts/character_select_team_bar_1024.json">
                Character select — team bar (6)
              </option>
              <option value="config/layouts/ingame_split_teams_1024.json">
                In-game YOUR TEAM vs ENEMY (12)
              </option>
            </select>
          </div>
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
              Drag &amp; Drop Image
            </p>
            <p className="mt-1 text-[10px] text-zinc-400 break-words">
              {imageLabelText}
            </p>
            <p className="mt-1 text-[9px] text-blue-200/80">
              Upload + ask in one flow: detected teams are carried into your next question.
            </p>
            {hasElectronIpc && (
              <p className="mt-2 text-[9px] text-zinc-500 leading-snug">
                Hotkey: ⌘⇧G (mac) or Ctrl+Shift+G captures underlay and runs the agent.
              </p>
            )}
          </label>
        </div>
      </aside>
    </div>
  );
}
