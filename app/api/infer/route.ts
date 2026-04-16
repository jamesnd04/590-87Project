import { existsSync } from "fs";
import { mkdir, readFile, unlink, writeFile } from "fs/promises";
import { NextRequest, NextResponse } from "next/server";
import path from "path";
import { runImageInferWithWorker } from "@/lib/infer/clip-worker";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const DEFAULT_LAYOUT = path.join(process.cwd(), "config", "scoreboard_layout.json");

const ROBOFLOW_SERVERLESS = "https://serverless.roboflow.com";
const DEFAULT_ROBOFLOW_WORKSPACE = "jamess-workspace-vyfdf";
const DEFAULT_ROBOFLOW_WORKFLOW_ID = "small-object-detection-sahi";
const DEFAULT_GEMINI_POST_GAME_MODEL = "gemini-2.0-flash";
const GEMINI_POST_GAME_PROMPT =
  "Analyze this post-game scoreboard from a Marvel Rivals match. Summarize how the game likely played out (team dynamics, win conditions, and why one team won). Evaluate my performance specifically (I am highlighted in yellow) in the context of my role. Infer he biggest mistakes or weaknesses from my team based on the scoreboard. Give 3–5 highly specific, actionable improvements I can apply in my next game (not generic advice). If possible, infer my role (tank/DPS/support) and suggest how I could have had more impact in that role. Be honest and critical, but focus on improvement and decision-making—not just stats.";

/** Remove huge base64 image strings from Roboflow JSON (keep type + length). */
function omitBase64ImageBlobs(value: unknown): unknown {
  if (value === null || value === undefined) {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map(omitBase64ImageBlobs);
  }
  if (typeof value === "object") {
    const o = value as Record<string, unknown>;
    if (o.type === "base64" && typeof o.value === "string") {
      return {
        ...o,
        value: `[omitted: ${o.value.length} base64 characters]`,
      };
    }
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(o)) {
      out[k] = omitBase64ImageBlobs(v);
    }
    return out;
  }
  return value;
}

type PredictionWithPlacement = {
  /** 0 = top row on that side of the scoreboard */
  row: number;
  side: "your_team_blue_left" | "enemy_team_right";
  class: string;
  /** Detection center x in pixels (Roboflow workflow convention). */
  x: number;
  y: number;
  width?: number;
  height?: number;
  confidence: number;
  class_id?: number;
  detection_id?: string;
};

function isDetectionLike(o: Record<string, unknown>): boolean {
  if (typeof o.class !== "string") {
    return false;
  }
  return (
    typeof o.confidence === "number" ||
    typeof o.class_id === "number" ||
    typeof o.detection_id === "string"
  );
}

/** Collect Roboflow detection objects that include bbox / center coordinates. */
function collectDetectionsWithGeometry(data: unknown): Array<Record<string, unknown>> {
  const out: Array<Record<string, unknown>> = [];

  function walk(node: unknown): void {
    if (node === null || node === undefined) {
      return;
    }
    if (Array.isArray(node)) {
      for (const item of node) {
        walk(item);
      }
      return;
    }
    if (typeof node === "object") {
      const o = node as Record<string, unknown>;
      if (
        isDetectionLike(o) &&
        typeof o.x === "number" &&
        typeof o.y === "number" &&
        typeof o.confidence === "number"
      ) {
        out.push(o);
      }
      for (const v of Object.values(o)) {
        walk(v);
      }
    }
  }

  walk(data);
  return out;
}

/**
 * Split detections into YOUR TEAM (left / blue UI) vs ENEMY (right) using x vs half image width,
 * then sort by y (top → bottom) and assign row indices per side.
 */
function splitPredictionsByTeam(
  raw: unknown,
  imageWidthPx: number,
): {
  image_width_px: number;
  split_x_px: number;
  note: string;
  your_team_blue_left: PredictionWithPlacement[];
  enemy_team_right: PredictionWithPlacement[];
} {
  const w = Number.isFinite(imageWidthPx) && imageWidthPx > 0 ? imageWidthPx : 1024;
  const mid = w / 2;
  const dets = collectDetectionsWithGeometry(raw);

  const left = dets.filter((d) => (d.x as number) < mid);
  const right = dets.filter((d) => (d.x as number) >= mid);

  const sortY = (a: Record<string, unknown>, b: Record<string, unknown>) =>
    (a.y as number) - (b.y as number);
  left.sort(sortY);
  right.sort(sortY);

  const mapSide = (
    rows: Record<string, unknown>[],
    side: PredictionWithPlacement["side"],
  ): PredictionWithPlacement[] =>
    rows.map((d, row) => ({
      row,
      side,
      class: d.class as string,
      x: d.x as number,
      y: d.y as number,
      width: typeof d.width === "number" ? d.width : undefined,
      height: typeof d.height === "number" ? d.height : undefined,
      confidence: d.confidence as number,
      class_id: typeof d.class_id === "number" ? d.class_id : undefined,
      detection_id: typeof d.detection_id === "string" ? d.detection_id : undefined,
    }));

  return {
    image_width_px: w,
    split_x_px: mid,
    note:
      "your_team_blue_left = detections with center x < half image width (YOUR TEAM column); enemy_team_right = x ≥ half width.",
    your_team_blue_left: mapSide(left, "your_team_blue_left"),
    enemy_team_right: mapSide(right, "enemy_team_right"),
  };
}

/** Flat class list: your team top→bottom, then enemy top→bottom (for simple consumers). */
function flattenTeamClasses(teams: ReturnType<typeof splitPredictionsByTeam>): string[] {
  return [
    ...teams.your_team_blue_left.map((p) => p.class),
    ...teams.enemy_team_right.map((p) => p.class),
  ];
}

/** Drop empty profiler noise from workflow JSON for clients. */
function stripWorkflowNoise(value: unknown): unknown {
  if (value === null || value === undefined) {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map(stripWorkflowNoise);
  }
  if (typeof value === "object") {
    const o = value as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(o)) {
      if (k === "profiler_trace" && Array.isArray(v) && v.length === 0) {
        continue;
      }
      out[k] = stripWorkflowNoise(v);
    }
    return out;
  }
  return value;
}

async function runRoboflowWorkflow(imagePath: string): Promise<Record<string, unknown>> {
  const apiKey = process.env.ROBOFLOW_API_KEY?.trim();
  if (!apiKey) {
    return { enabled: false, skipped: "ROBOFLOW_API_KEY not set" };
  }
  const workspace = process.env.ROBOFLOW_WORKSPACE_NAME?.trim() || DEFAULT_ROBOFLOW_WORKSPACE;
  const workflowId = process.env.ROBOFLOW_WORKFLOW_ID?.trim() || DEFAULT_ROBOFLOW_WORKFLOW_ID;
  const url = `${ROBOFLOW_SERVERLESS}/${workspace}/workflows/${workflowId}`;
  let buf: Buffer;
  try {
    buf = await readFile(imagePath);
  } catch (e) {
    return {
      enabled: true,
      ok: false,
      workspace,
      workflow_id: workflowId,
      error: e instanceof Error ? e.message : "Could not read image for Roboflow",
    };
  }
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        api_key: apiKey,
        use_cache: true,
        inputs: {
          image: { type: "base64", value: buf.toString("base64") },
        },
      }),
    });
    const text = await res.text();
    let data: unknown;
    try {
      data = JSON.parse(text) as unknown;
    } catch {
      return {
        enabled: true,
        ok: false,
        workspace,
        workflow_id: workflowId,
        http_status: res.status,
        error: "Roboflow response was not JSON",
        raw: text.slice(0, 2000),
      };
    }
    if (!res.ok) {
      return {
        enabled: true,
        ok: false,
        workspace,
        workflow_id: workflowId,
        http_status: res.status,
        error:
          typeof data === "object" && data !== null && "message" in data
            ? String((data as { message: unknown }).message)
            : `HTTP ${res.status}`,
        body: data,
      };
    }
    return {
      enabled: true,
      ok: true,
      workspace,
      workflow_id: workflowId,
      result: data,
    };
  } catch (e) {
    return {
      enabled: true,
      ok: false,
      workspace,
      workflow_id: workflowId,
      error: e instanceof Error ? e.message : String(e),
    };
  }
}

async function runGeminiPostGameAnalysis(imagePath: string): Promise<{
  ok: boolean;
  text?: string;
  model?: string;
  error?: string;
}> {
  const apiKey = process.env.GEMINI_API_KEY?.trim();
  if (!apiKey) {
    return { ok: false, error: "GEMINI_API_KEY not set" };
  }
  let imageBuffer: Buffer;
  try {
    imageBuffer = await readFile(imagePath);
  } catch (e) {
    return {
      ok: false,
      error: e instanceof Error ? e.message : "Could not read image for Gemini",
    };
  }
  const model = process.env.GEMINI_POST_GAME_MODEL?.trim() || DEFAULT_GEMINI_POST_GAME_MODEL;
  const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;
  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-goog-api-key": apiKey,
      },
      body: JSON.stringify({
        contents: [
          {
            parts: [
              { text: GEMINI_POST_GAME_PROMPT },
              {
                inline_data: {
                  mime_type: "image/png",
                  data: imageBuffer.toString("base64"),
                },
              },
            ],
          },
        ],
      }),
    });
    const data = (await res.json()) as {
      candidates?: Array<{
        content?: {
          parts?: Array<{ text?: unknown }>;
        };
      }>;
      error?: { message?: unknown };
    };
    if (!res.ok) {
      const errMsg =
        typeof data?.error?.message === "string" ? data.error.message : `HTTP ${res.status}`;
      return { ok: false, model, error: errMsg };
    }
    const textParts: string[] = [];
    for (const candidate of data.candidates ?? []) {
      for (const part of candidate.content?.parts ?? []) {
        if (typeof part.text === "string" && part.text.trim()) {
          textParts.push(part.text.trim());
        }
      }
    }
    const text = textParts.join("\n\n").trim();
    if (!text) {
      return { ok: false, model, error: "Gemini returned no text" };
    }
    return { ok: true, model, text };
  } catch (e) {
    return {
      ok: false,
      model,
      error: e instanceof Error ? e.message : "Gemini request failed",
    };
  }
}

function runImageInfer(imagePath: string, layoutPath: string | null): Promise<{ stdout: string; stderr: string; code: number | null }> {
  return runImageInferWithWorker(imagePath, layoutPath);
}

export async function POST(req: NextRequest) {
  let form: FormData;
  try {
    form = await req.formData();
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid form data" }, { status: 400 });
  }

  const file = form.get("image");
  if (!file || !(file instanceof Blob)) {
    return NextResponse.json({ ok: false, error: "Missing image field (expected name: image)" }, { status: 400 });
  }

  const skipRoboflowField = form.get("skip_roboflow");
  const skipRoboflow =
    skipRoboflowField === null ||
    skipRoboflowField === "1" ||
    skipRoboflowField === "true";

  const layoutField = form.get("layout");
  let layoutPath: string | null = null;
  if (typeof layoutField === "string" && layoutField.trim()) {
    const resolved = path.isAbsolute(layoutField)
      ? layoutField
      : path.join(process.cwd(), layoutField.trim());
    if (!existsSync(resolved)) {
      return NextResponse.json(
        { ok: false, error: `Layout file not found: ${resolved}` },
        { status: 400 },
      );
    }
    layoutPath = resolved;
  } else if (existsSync(DEFAULT_LAYOUT)) {
    layoutPath = DEFAULT_LAYOUT;
  }

  const tmpRoot = path.join(process.cwd(), ".tmp-infer");
  await mkdir(tmpRoot, { recursive: true });
  const ext = guessExt(file);
  const tmpPath = path.join(tmpRoot, `upload-${Date.now()}${ext}`);
  const buf = Buffer.from(await file.arrayBuffer());
  await writeFile(tmpPath, buf);

  try {
    const { stdout, stderr, code } = await runImageInfer(tmpPath, layoutPath);

    if (code !== 0) {
      const msg =
        stderr.trim() ||
        stdout.trim() ||
        `Python exited with code ${code}. Ensure dependencies are installed: pip install -r requirements-ml.txt`;
      return NextResponse.json(
        { ok: false, error: msg, stderr: stderr.trim() || undefined },
        { status: 500 },
      );
    }

    const line = stdout
      .trim()
      .split("\n")
      .filter(Boolean)
      .pop();
    if (!line) {
      return NextResponse.json(
        { ok: false, error: "Empty output from inference script", stderr: stderr.trim() || undefined },
        { status: 500 },
      );
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(line) as Record<string, unknown>;
    } catch {
      return NextResponse.json(
        {
          ok: false,
          error: "Inference did not return valid JSON",
          raw: stdout.slice(0, 4000),
          stderr: stderr.trim() || undefined,
        },
        { status: 500 },
      );
    }

    const rec = parsed as { ok?: boolean; summary?: string; error?: string };
    if (rec.ok === false && rec.error) {
      return NextResponse.json({ ok: false, error: String(rec.error) }, { status: 500 });
    }

    const payload = parsed as Record<string, unknown>;
    let summaryText = typeof rec.summary === "string" ? rec.summary : "";
    const layoutObj =
      typeof payload.layout === "object" && payload.layout !== null
        ? (payload.layout as Record<string, unknown>)
        : null;
    const layoutMode =
      typeof layoutObj?.layout_mode === "string"
        ? String(layoutObj.layout_mode).toLowerCase()
        : "";
    const isPostGameLayout = layoutMode === "post_game";
    const isPreGameLayout = layoutMode === "pre_game";

    if (isPostGameLayout) {
      const gemini = await runGeminiPostGameAnalysis(tmpPath);
      if (!gemini.ok || !gemini.text) {
        return NextResponse.json(
          { ok: false, error: `Post-game Gemini analysis failed: ${String(gemini.error ?? "unknown error")}` },
          { status: 500 },
        );
      }
      return NextResponse.json({
        ok: true,
        summary: gemini.text,
        analysis: gemini.text,
        prediction_classes: [],
        prediction_teams: null,
        payload: {
          ...payload,
          gemini: {
            provider: "google",
            model: gemini.model,
            prompt: GEMINI_POST_GAME_PROMPT,
            text: gemini.text,
          },
        },
      });
    }

    const initialPredictionClasses = Array.isArray(payload.prediction_classes)
      ? payload.prediction_classes.filter((v): v is string => typeof v === "string")
      : [];
    let predictionClasses: string[] = [...initialPredictionClasses];
    let predictionTeams: ReturnType<typeof splitPredictionsByTeam> | null = null;

    const imageMeta = payload.image as { width?: unknown } | undefined;
    const imageWidthPx =
      typeof imageMeta?.width === "number" && Number.isFinite(imageMeta.width)
        ? imageMeta.width
        : 1024;

    if (!skipRoboflow) {
      const roboflow = await runRoboflowWorkflow(tmpPath);
      if (roboflow.enabled && roboflow.ok === true && roboflow.result !== undefined) {
        const withoutBlobs = omitBase64ImageBlobs(roboflow.result) as unknown;
        const trimmed = stripWorkflowNoise(withoutBlobs) as unknown;
        predictionTeams = splitPredictionsByTeam(roboflow.result, imageWidthPx);
        if (isPreGameLayout) {
          const mergedRows = [
            ...predictionTeams.your_team_blue_left,
            ...predictionTeams.enemy_team_right,
          ]
            .sort((a, b) => a.y - b.y)
            .map((p, row) => ({
              ...p,
              row,
              side: "your_team_blue_left" as const,
            }));
          predictionTeams = {
            ...predictionTeams,
            note:
              "pre_game layout: enemy team slots are not present; all identified classes are assigned to your_team_blue_left.",
            your_team_blue_left: mergedRows,
            enemy_team_right: [],
          };
        }
        predictionClasses = flattenTeamClasses(predictionTeams);
        console.log(
          "[infer] YOUR TEAM (blue, left column) — predictions top→bottom:",
          predictionTeams.your_team_blue_left,
        );
        console.log(
          "[infer] ENEMY TEAM (right column) — predictions top→bottom:",
          predictionTeams.enemy_team_right,
        );
        console.log("[infer] prediction_classes (your team rows, then enemy rows):", predictionClasses);
        console.log("[infer] Roboflow workflow result (base64 JPEGs omitted):");
        console.log(JSON.stringify(trimmed, null, 2));
        payload.roboflow = {
          ...roboflow,
          result: trimmed,
          prediction_classes: predictionClasses,
          prediction_teams: predictionTeams,
        };
        summaryText = [summaryText, "Roboflow serverless workflow (SAHI): OK."]
          .filter(Boolean)
          .join("\n");
      } else {
        if (roboflow.enabled) {
          console.log("[infer] Roboflow (non-success or no result):", JSON.stringify(roboflow, null, 2));
        }
        payload.roboflow = roboflow;
        if (roboflow.enabled && roboflow.ok === false) {
          summaryText = [summaryText, `Roboflow workflow failed: ${String(roboflow.error ?? "unknown")}`]
            .filter(Boolean)
            .join("\n");
        }
      }
    } else {
      payload.roboflow = { enabled: false, skipped: "skip_roboflow form field set" };
    }

    payload.prediction_classes = predictionClasses;
    if (predictionTeams) {
      payload.prediction_teams = predictionTeams;
    }

    return NextResponse.json({
      ok: true,
      summary: summaryText,
      prediction_classes: predictionClasses,
      prediction_teams: predictionTeams,
      payload,
    });
  } finally {
    try {
      await unlink(tmpPath);
    } catch {
      /* ignore */
    }
  }
}

function guessExt(file: Blob & { name?: string }) {
  const n = typeof file.name === "string" ? file.name.toLowerCase() : "";
  if (n.endsWith(".png")) return ".png";
  if (n.endsWith(".jpg") || n.endsWith(".jpeg")) return ".jpg";
  if (n.endsWith(".webp")) return ".webp";
  if (n.endsWith(".bmp")) return ".bmp";
  const t = file.type;
  if (t === "image/png") return ".png";
  if (t === "image/jpeg") return ".jpg";
  if (t === "image/webp") return ".webp";
  if (t === "image/bmp") return ".bmp";
  return ".png";
}
