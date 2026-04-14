import { spawn } from "child_process";
import { existsSync } from "fs";
import path from "path";

export function pythonBinary() {
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

export function runImageInfer(
  imagePath: string,
  layoutPath: string | null,
): Promise<{ stdout: string; stderr: string; code: number | null }> {
  const script = path.join(process.cwd(), "scripts", "image_infer.py");
  const argv = [script, "--source", imagePath];
  if (layoutPath) {
    argv.push("--layout", layoutPath);
  }

  return new Promise((resolve) => {
    const child = spawn(pythonBinary(), argv, {
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });
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

export function guessExt(file: Blob & { name?: string }) {
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
