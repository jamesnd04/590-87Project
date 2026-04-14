"use client";

import { useCallback, useMemo, useState } from "react";

export default function Home() {
  const [droppedImages, setDroppedImages] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [aiText, setAiText] = useState<string | null>(null);
  const [inferStatus, setInferStatus] = useState<"idle" | "loading" | "error">("idle");
  const [inferError, setInferError] = useState<string | null>(null);
  const [layoutPreset, setLayoutPreset] = useState<string>("");

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
      if (layoutPreset.trim()) {
        body.append("layout", layoutPreset.trim());
      }
      const res = await fetch("/api/infer", { method: "POST", body });
      const data = (await res.json()) as {
        ok?: boolean;
        summary?: string;
        error?: string;
        stderr?: string;
        prediction_classes?: string[];
        prediction_teams?: {
          note?: string;
          image_width_px?: number;
          split_x_px?: number;
          your_team_blue_left?: unknown[];
          enemy_team_right?: unknown[];
        } | null;
        payload?: unknown;
      };
      if (!res.ok || !data.ok) {
        setInferStatus("error");
        setInferError(data.error ?? `Request failed (${res.status})`);
        return;
      }
      const classes = Array.isArray(data.prediction_classes) ? data.prediction_classes : [];
      const teams = data.prediction_teams;
      const yourTeam = Array.isArray(teams?.your_team_blue_left) ? teams!.your_team_blue_left : [];
      const enemyTeam = Array.isArray(teams?.enemy_team_right) ? teams!.enemy_team_right : [];

      console.log("[infer] YOUR TEAM (blue, left column) — top→bottom:", yourTeam);
      console.log("[infer] ENEMY TEAM (right column) — top→bottom:", enemyTeam);
      console.log("[infer] prediction_classes (your rows, then enemy):", classes);

      const summary = data.summary?.trim() ?? "";
      const pretty =
        data.payload !== undefined && data.payload !== null
          ? JSON.stringify(data.payload, null, 2)
          : "";

      const teamMeta =
        teams && typeof teams.image_width_px === "number" && typeof teams.split_x_px === "number"
          ? `Split at x=${teams.split_x_px}px (image width ${teams.image_width_px}px). ${teams.note ?? ""}`
          : teams?.note ?? "";

      const yourBlock =
        yourTeam.length > 0
          ? `YOUR TEAM (blue, left)\n${JSON.stringify(yourTeam, null, 2)}`
          : "YOUR TEAM (blue, left): (no detections with x/y on this side)";
      const enemyBlock =
        enemyTeam.length > 0
          ? `ENEMY TEAM (right)\n${JSON.stringify(enemyTeam, null, 2)}`
          : "ENEMY TEAM (right): (no detections with x/y on this side)";

      const classesBlock =
        classes.length > 0
          ? `All class names (your team rows, then enemy rows):\n${JSON.stringify(classes, null, 2)}`
          : "All class names: []";

      const teamsSection =
        teams !== undefined && teams !== null
          ? [teamMeta && `--- Team split ---\n${teamMeta}`, yourBlock, enemyBlock, classesBlock]
              .filter(Boolean)
              .join("\n\n")
          : classesBlock;

      setAiText(
        [teamsSection, summary, pretty ? `\n--- Full payload JSON ---\n${pretty}` : ""]
          .filter(Boolean)
          .join("\n\n"),
      );
      setInferStatus("idle");
    } catch (e) {
      setInferStatus("error");
      setInferError(e instanceof Error ? e.message : "Inference request failed");
    }
  }, [layoutPreset]);

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
                Extracting avatar regions (OpenCV layout)…
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
                Drop a screenshot to extract icon boxes. Default layout:{" "}
                <span className="text-zinc-400">config/scoreboard_layout.json</span> (12-row
                post-match). Split-screen vs match:{" "}
                <span className="text-zinc-400">config/layouts/ingame_split_teams_1024.json</span>.
                Other presets: team panel, character select bar (form field{" "}
                <span className="text-zinc-400">layout</span>).
              </p>
            )}
          </div>
        </div>

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
