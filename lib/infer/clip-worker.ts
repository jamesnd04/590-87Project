import { spawn, type ChildProcessWithoutNullStreams } from "child_process";
import path from "path";

const RESPONSE_PREFIX = "__CLIP_WORKER_JSON__";

type InferRequest = {
  source: string;
  layout?: string | null;
  normalize_size?: number;
  hero_assets?: string;
  clip_model?: string;
  clip_top_k?: number;
  clip_margin_threshold?: number;
  clip_require_confidence?: boolean;
  clip_include_lord_refs?: boolean;
  clip_cache_file?: string;
};

type Pending = {
  resolve: (value: {
    stdout: string;
    stderr: string;
    code: number | null;
  }) => void;
  timer: NodeJS.Timeout;
};

let worker: ChildProcessWithoutNullStreams | null = null;
let stderrBuffer = "";
let stdoutBuffer = "";
const pending = new Map<string, Pending>();

function pythonBinary() {
  const unixVenv = path.join(process.cwd(), ".venv", "bin", "python3");
  const winVenv = path.join(process.cwd(), ".venv", "Scripts", "python.exe");
  const { existsSync } = require("fs") as typeof import("fs");
  if (existsSync(unixVenv)) return unixVenv;
  if (existsSync(winVenv)) return winVenv;
  return process.platform === "win32" ? "python" : "python3";
}

function buildRequestFromEnv(
  source: string,
  layout: string | null,
): InferRequest {
  const req: InferRequest = { source, layout: layout ?? null, clip_top_k: 5 };
  if (process.env.CLIP_MODEL?.trim())
    req.clip_model = process.env.CLIP_MODEL.trim();
  if (process.env.CLIP_TOP_K?.trim())
    req.clip_top_k = Number(process.env.CLIP_TOP_K.trim());
  if (process.env.CLIP_MARGIN_THRESHOLD?.trim())
    req.clip_margin_threshold = Number(
      process.env.CLIP_MARGIN_THRESHOLD.trim(),
    );
  if (process.env.HERO_ASSETS_FILE?.trim())
    req.hero_assets = process.env.HERO_ASSETS_FILE.trim();
  if (/^(1|true|yes|on)$/i.test(process.env.CLIP_REQUIRE_CONFIDENCE || ""))
    req.clip_require_confidence = true;
  if (/^(1|true|yes|on)$/i.test(process.env.CLIP_INCLUDE_LORD_REFS || ""))
    req.clip_include_lord_refs = true;
  if (process.env.CLIP_CACHE_FILE?.trim())
    req.clip_cache_file = process.env.CLIP_CACHE_FILE.trim();
  return req;
}

function ensureWorker(): ChildProcessWithoutNullStreams {
  if (worker && !worker.killed) {
    return worker;
  }
  const script = path.join(process.cwd(), "scripts", "image_infer.py");
  const child = spawn(pythonBinary(), [script, "--worker"], {
    env: {
      ...process.env,
      PYTHONUNBUFFERED: "1",
      HF_HOME:
        process.env.HF_HOME ||
        path.join(process.cwd(), ".cache", "huggingface"),
    },
    stdio: ["pipe", "pipe", "pipe"],
  });
  worker = child;
  stderrBuffer = "";
  stdoutBuffer = "";

  child.stderr.on("data", (d: Buffer) => {
    stderrBuffer += d.toString();
    if (stderrBuffer.length > 20_000) {
      stderrBuffer = stderrBuffer.slice(-20_000);
    }
  });

  child.stdout.on("data", (d: Buffer) => {
    stdoutBuffer += d.toString();
    const lines = stdoutBuffer.split("\n");
    stdoutBuffer = lines.pop() || "";
    for (const line of lines) {
      const cleaned = line.trim();
      if (!cleaned) {
        continue;
      }
      if (!cleaned.startsWith(RESPONSE_PREFIX)) {
        continue;
      }
      try {
        const payload = JSON.parse(cleaned.slice(RESPONSE_PREFIX.length)) as {
          id: string;
          ok: boolean;
          result?: unknown;
          error?: string;
        };
        const req = pending.get(payload.id);
        if (!req) continue;
        clearTimeout(req.timer);
        pending.delete(payload.id);
        if (payload.ok && payload.result) {
          req.resolve({
            stdout: JSON.stringify(payload.result),
            stderr: stderrBuffer,
            code: 0,
          });
        } else {
          req.resolve({
            stdout: "",
            stderr: payload.error || stderrBuffer || "Worker error",
            code: 1,
          });
        }
      } catch {
        // ignore malformed worker line
      }
    }
  });

  const teardown = () => {
    for (const [id, req] of pending.entries()) {
      clearTimeout(req.timer);
      req.resolve({
        stdout: "",
        stderr: stderrBuffer || "CLIP worker exited unexpectedly",
        code: 1,
      });
      pending.delete(id);
    }
    worker = null;
  };
  child.on("exit", teardown);
  child.on("error", teardown);

  return child;
}

export async function runImageInferWithWorker(
  imagePath: string,
  layoutPath: string | null,
): Promise<{ stdout: string; stderr: string; code: number | null }> {
  try {
    const child = ensureWorker();
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    const payload = buildRequestFromEnv(imagePath, layoutPath);
    return await new Promise((resolve) => {
      const timer = setTimeout(() => {
        pending.delete(id);
        resolve({
          stdout: "",
          stderr: "Timed out waiting for CLIP worker response",
          code: 1,
        });
      }, 120_000);
      pending.set(id, { resolve, timer });
      child.stdin.write(`${JSON.stringify({ id, payload })}\n`);
    });
  } catch (e) {
    return {
      stdout: "",
      stderr: e instanceof Error ? e.message : String(e),
      code: 1,
    };
  }
}
