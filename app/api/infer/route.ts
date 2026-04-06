import { spawn } from "child_process";
import { existsSync } from "fs";
import { mkdir, unlink, writeFile } from "fs/promises";
import { NextRequest, NextResponse } from "next/server";
import path from "path";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const SUITE_PT = path.join(
  process.cwd(),
  "models",
  "MarvelRivals-Detection-Suite",
  "pt",
);

function modelPaths() {
  return {
    model1: path.join(SUITE_PT, "hero.pt"),
    model2: path.join(SUITE_PT, "hp.pt"),
    model3: path.join(SUITE_PT, "ui.pt"),
    model4: path.join(SUITE_PT, "friendfoe.pt"),
  };
}

function pythonBinary() {
  const unixVenv = path.join(process.cwd(), ".venv", "bin", "python3");
  if (existsSync(unixVenv)) {
    return unixVenv;
  }
  const winVenv = path.join(process.cwd(), ".venv", "Scripts", "python.exe");
  if (existsSync(winVenv)) {
    return winVenv;
  }
  return process.platform === "win32" ? "python" : "python3";
}

function runImageInfer(imagePath: string): Promise<{ stdout: string; stderr: string; code: number | null }> {
  const script = path.join(process.cwd(), "scripts", "image_infer.py");
  const { model1, model2, model3, model4 } = modelPaths();
  const bin = pythonBinary();

  return new Promise((resolve) => {
    const child = spawn(
      bin,
      [
        script,
        "--model1",
        model1,
        "--model2",
        model2,
        "--model3",
        model3,
        "--model4",
        model4,
        "--source",
        imagePath,
        "--thresh",
        "0.25",
      ],
      {
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
      },
    );
    let stdout = "";
    let stderr = "";
    child.stdout?.on("data", (d: Buffer) => {
      stdout += d.toString();
    });
    child.stderr?.on("data", (d: Buffer) => {
      stderr += d.toString();
    });
    child.on("close", (code) => {
      resolve({ stdout, stderr, code });
    });
    child.on("error", (err) => {
      stderr += err.message;
      resolve({ stdout, stderr, code: -1 });
    });
  });
}

export async function POST(req: NextRequest) {
  const m = modelPaths();
  for (const p of Object.values(m)) {
    if (!existsSync(p)) {
      return NextResponse.json(
        {
          ok: false,
          error:
            "Model weights not found. From the project root, run: hf download Chappieut/MarvelRivals-Detection-Suite --local-dir models/MarvelRivals-Detection-Suite",
        },
        { status: 503 },
      );
    }
  }

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

  const tmpRoot = path.join(process.cwd(), ".tmp-infer");
  await mkdir(tmpRoot, { recursive: true });
  const ext = guessExt(file);
  const tmpPath = path.join(tmpRoot, `upload-${Date.now()}${ext}`);
  const buf = Buffer.from(await file.arrayBuffer());
  await writeFile(tmpPath, buf);

  try {
    const { stdout, stderr, code } = await runImageInfer(tmpPath);

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

    return NextResponse.json({
      ok: true,
      summary: typeof rec.summary === "string" ? rec.summary : "",
      payload: parsed,
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
