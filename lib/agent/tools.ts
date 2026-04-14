import { spawn } from "child_process";
import { existsSync } from "fs";
import { mkdir, readFile, unlink, writeFile } from "fs/promises";
import path from "path";

const STRATEGY_PATHS = [
  path.join(process.cwd(), "marvel_rivals_strategy.md"),
  path.join(process.cwd(), "marvel_rivals_strategy_guide.md"),
];

type Intent = "image_analysis" | "strategy_question" | "external_api_lookup" | "general_chat";

export type ToolUsage = {
  tool: string;
  ok: boolean;
  details?: string;
};

export type AgentResult = {
  intent: Intent;
  tools: ToolUsage[];
  response: string;
  context: string[];
};

type YoloResult = {
  ok: boolean;
  predictionClasses: string[];
  summary: string;
  raw?: Record<string, unknown>;
  error?: string;
};

function detectIntent(input: string, hasImage: boolean): Intent {
  if (hasImage) {
    return "image_analysis";
  }
  const normalized = input.toLowerCase();
  if (/(api|endpoint|external|live data|fetch data|stats)/.test(normalized)) {
    return "external_api_lookup";
  }
  if (/(how|what|why|guide|strategy|pick|counter|team|hero|comp)/.test(normalized)) {
    return "strategy_question";
  }
  return "general_chat";
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

function runImageInfer(imagePath: string, layoutPath: string | null): Promise<{ stdout: string; stderr: string; code: number | null }> {
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

function parseYoloClasses(raw: Record<string, unknown>): string[] {
  const rf = raw.roboflow as Record<string, unknown> | undefined;
  const classes = rf?.prediction_classes;
  if (Array.isArray(classes)) {
    return classes.filter((item): item is string => typeof item === "string");
  }
  const fallback = raw.prediction_classes;
  if (Array.isArray(fallback)) {
    return fallback.filter((item): item is string => typeof item === "string");
  }
  return [];
}

export async function runYoloTool(image: Blob, layoutPreset: string): Promise<YoloResult> {
  const tmpRoot = path.join(process.cwd(), ".tmp-infer");
  await mkdir(tmpRoot, { recursive: true });
  const tmpPath = path.join(tmpRoot, `agent-upload-${Date.now()}${guessExt(image as Blob & { name?: string })}`);
  const buf = Buffer.from(await image.arrayBuffer());
  await writeFile(tmpPath, buf);

  const layoutPath = layoutPreset.trim() ? path.join(process.cwd(), layoutPreset.trim()) : null;

  try {
    const { stdout, stderr, code } = await runImageInfer(tmpPath, layoutPath);
    if (code !== 0) {
      return {
        ok: false,
        predictionClasses: [],
        summary: "",
        error: stderr.trim() || stdout.trim() || `YOLO pipeline exited with code ${code}`,
      };
    }

    const line = stdout
      .trim()
      .split("\n")
      .filter(Boolean)
      .pop();
    if (!line) {
      return { ok: false, predictionClasses: [], summary: "", error: "YOLO output was empty" };
    }

    const parsed = JSON.parse(line) as Record<string, unknown>;
    return {
      ok: true,
      predictionClasses: parseYoloClasses(parsed),
      summary: typeof parsed.summary === "string" ? parsed.summary : "YOLO completed.",
      raw: parsed,
    };
  } catch (error) {
    return {
      ok: false,
      predictionClasses: [],
      summary: "",
      error: error instanceof Error ? error.message : "Failed to run YOLO tool",
    };
  } finally {
    try {
      await unlink(tmpPath);
    } catch {
      // Ignore cleanup errors.
    }
  }
}

type StrategyChunk = {
  heading: string;
  body: string;
  fullText: string;
  tokens: string[];
};

type ScoredChunk = {
  chunk: StrategyChunk;
  score: number;
  overlap: number;
};

const STOP_WORDS = new Set([
  "the",
  "a",
  "an",
  "and",
  "or",
  "for",
  "to",
  "of",
  "in",
  "on",
  "is",
  "are",
  "be",
  "as",
  "at",
  "by",
  "it",
  "with",
  "from",
  "that",
  "this",
  "your",
  "you",
  "we",
  "our",
  "they",
  "their",
  "can",
  "should",
  "into",
  "when",
  "what",
  "how",
  "why",
  "which",
  "who",
]);

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/g)
    .filter((term) => term.length > 2 && !STOP_WORDS.has(term));
}

function splitStrategyChunks(content: string): StrategyChunk[] {
  const lines = content.split("\n");
  const chunks: StrategyChunk[] = [];
  let currentHeading = "General";
  let currentLines: string[] = [];

  const flush = () => {
    const body = currentLines.join("\n").trim();
    if (!body) {
      return;
    }
    const fullText = `${currentHeading}\n${body}`.trim();
    chunks.push({
      heading: currentHeading,
      body,
      fullText,
      tokens: tokenize(fullText),
    });
  };

  for (const line of lines) {
    if (/^#{2,4}\s+/.test(line)) {
      flush();
      currentHeading = line.replace(/^#{2,4}\s+/, "").trim();
      currentLines = [];
      continue;
    }
    currentLines.push(line);
  }
  flush();

  // Fallback for malformed markdown.
  if (chunks.length === 0) {
    const body = content.trim();
    if (body) {
      chunks.push({
        heading: "General",
        body,
        fullText: body,
        tokens: tokenize(body),
      });
    }
  }

  return chunks;
}

function keywordFrequency(tokens: string[]): Map<string, number> {
  const out = new Map<string, number>();
  for (const token of tokens) {
    out.set(token, (out.get(token) || 0) + 1);
  }
  return out;
}

function scoreChunk(chunk: StrategyChunk, query: string, queryTokens: string[]): ScoredChunk {
  const querySet = new Set(queryTokens);
  const chunkFreq = keywordFrequency(chunk.tokens);
  let overlap = 0;
  let lexicalScore = 0;

  for (const token of querySet) {
    const tf = chunkFreq.get(token) || 0;
    if (tf > 0) {
      overlap += 1;
      lexicalScore += 1 + Math.min(tf, 3) * 0.35;
    }
  }

  const headingBoost = queryTokens.some((token) => chunk.heading.toLowerCase().includes(token))
    ? 1.5
    : 0;
  const phraseBoost =
    query.trim().length > 5 && chunk.fullText.toLowerCase().includes(query.trim().toLowerCase())
      ? 2.2
      : 0;
  const density = overlap > 0 ? overlap / Math.max(6, chunk.tokens.length / 14) : 0;
  const score = lexicalScore + headingBoost + phraseBoost + density;

  return {
    chunk,
    score,
    overlap,
  };
}

function jaccardSimilarity(a: string[], b: string[]): number {
  const setA = new Set(a);
  const setB = new Set(b);
  let intersection = 0;
  for (const token of setA) {
    if (setB.has(token)) {
      intersection += 1;
    }
  }
  const union = setA.size + setB.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

function takeDiverseTop(scored: ScoredChunk[], topK: number): ScoredChunk[] {
  const selected: ScoredChunk[] = [];
  for (const candidate of scored) {
    if (selected.length >= topK) {
      break;
    }
    const tooSimilar = selected.some(
      (picked) => jaccardSimilarity(candidate.chunk.tokens, picked.chunk.tokens) > 0.78,
    );
    if (!tooSimilar) {
      selected.push(candidate);
    }
  }
  return selected;
}

function trimSnippet(text: string, maxChars: number): string {
  if (text.length <= maxChars) {
    return text;
  }
  return `${text.slice(0, maxChars).trimEnd()}...`;
}

export async function searchStrategyTool(query: string): Promise<string[]> {
  const strategyPath = STRATEGY_PATHS.find((candidate) => existsSync(candidate));
  if (!strategyPath) {
    return ["No strategy guide file found. Expected marvel_rivals_strategy.md or marvel_rivals_strategy_guide.md."];
  }

  const content = await readFile(strategyPath, "utf-8");
  const chunks = splitStrategyChunks(content);

  const queryTokens = tokenize(query);
  const ranked = chunks
    .map((chunk) => scoreChunk(chunk, query, queryTokens))
    .filter((item) => item.score > 0 || queryTokens.length === 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 12);

  const diverseTop = takeDiverseTop(ranked, 4);

  if (diverseTop.length > 0) {
    return diverseTop.map(({ chunk, overlap, score }) =>
      `[${chunk.heading}] score=${score.toFixed(2)} overlap=${overlap}\n${trimSnippet(
        chunk.body,
        700,
      )}`,
    );
  }

  return chunks.slice(0, 2).map((chunk) => `[${chunk.heading}]\n${trimSnippet(chunk.body, 700)}`);
}

export async function callFutureApiTool(_query: string): Promise<{ ok: boolean; result: string }> {
  // Placeholder for the upcoming external API integration.
  return {
    ok: false,
    result: "TODO: external API tool not implemented yet.",
  };
}

async function callOpenAIResponse(prompt: string): Promise<string | null> {
  const apiKey = process.env.OPENAI_API_KEY?.trim();
  if (!apiKey) {
    return null;
  }

  const model = process.env.OPENAI_MODEL?.trim() || "gpt-4.1-mini";
  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages: [
        {
          role: "system",
          content:
            "You are a Marvel Rivals strategy agent. Use the provided tool context before answering. If context is missing, state assumptions briefly.",
        },
        {
          role: "user",
          content: prompt,
        },
      ],
      temperature: 0.3,
    }),
  });

  if (!response.ok) {
    return null;
  }
  const data = (await response.json()) as {
    choices?: Array<{ message?: { content?: string } }>;
  };
  return data.choices?.[0]?.message?.content?.trim() || null;
}

function fallbackResponse(userInput: string, context: string[]): string {
  const joinedContext = context.length > 0 ? context.join("\n\n") : "No extra context was found.";
  return `I analyzed your request: "${userInput || "(no text provided)"}"\n\n${joinedContext}`;
}

export async function runAgent(input: { userInput: string; image: Blob | null; layoutPreset: string }): Promise<AgentResult> {
  const intent = detectIntent(input.userInput, Boolean(input.image));
  const tools: ToolUsage[] = [];
  const context: string[] = [];

  if (input.image) {
    const yolo = await runYoloTool(input.image, input.layoutPreset);
    tools.push({
      tool: "runYoloTool",
      ok: yolo.ok,
      details: yolo.ok
        ? `Detected ${yolo.predictionClasses.length} class entries.`
        : yolo.error || "YOLO failed",
    });
    if (yolo.ok) {
      const preview = yolo.predictionClasses.slice(0, 12).join(", ");
      context.push(
        `YOLO summary: ${yolo.summary}\nDetected classes: ${preview || "(none detected)"}`,
      );
    }
  }

  if (intent === "strategy_question" || input.userInput.trim().length > 0) {
    const snippets = await searchStrategyTool(input.userInput);
    tools.push({
      tool: "searchStrategyTool",
      ok: snippets.length > 0,
      details: `Retrieved ${snippets.length} strategy snippets.`,
    });
    context.push(`Strategy snippets:\n${snippets.map((s, i) => `[${i + 1}] ${s}`).join("\n\n")}`);
  }

  if (intent === "external_api_lookup") {
    const apiResult = await callFutureApiTool(input.userInput);
    tools.push({
      tool: "callFutureApiTool",
      ok: apiResult.ok,
      details: apiResult.result,
    });
    context.push(`External API tool: ${apiResult.result}`);
  }

  const prompt = [
    `User intent: ${intent}`,
    `User input: ${input.userInput || "(empty)"}`,
    "Tool context:",
    context.join("\n\n"),
  ]
    .filter(Boolean)
    .join("\n\n");

  const llm = await callOpenAIResponse(prompt);
  return {
    intent,
    tools,
    context,
    response: llm || fallbackResponse(input.userInput, context),
  };
}
