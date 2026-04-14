import { spawn } from "child_process";
import { existsSync } from "fs";
import { mkdir, readFile, unlink, writeFile } from "fs/promises";
import path from "path";

const STRATEGY_PATHS = [
  path.join(process.cwd(), "marvel_rivals_strategy.md"),
  path.join(process.cwd(), "marvel_rivals_strategy_guide.md"),
];
let strategyVocabularyCache: Set<string> | null = null;

type Intent =
  | "image_analysis"
  | "strategy_question"
  | "relationship_question"
  | "external_api_lookup"
  | "general_chat";

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
  imageContext?: ImageContext;
};

export type ImageContext = {
  yourTeam: string[];
  enemyTeam: string[];
  source: "image_upload" | "carried_context";
};

type YoloResult = {
  ok: boolean;
  predictionClasses: string[];
  summary: string;
  raw?: Record<string, unknown>;
  error?: string;
  imageContext?: ImageContext;
};

function detectIntent(input: string, hasImage: boolean): Intent {
  if (hasImage) {
    return "image_analysis";
  }
  const normalized = input.toLowerCase();
  if (isRelationshipQuestion(normalized)) {
    return "relationship_question";
  }
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

function splitTeamClasses(predictionClasses: string[]): ImageContext {
  const clean = predictionClasses.map((name) => name.trim()).filter(Boolean);
  if (clean.length === 0) {
    return {
      yourTeam: [],
      enemyTeam: [],
      source: "image_upload",
    };
  }
  const mid = Math.ceil(clean.length / 2);
  return {
    yourTeam: clean.slice(0, mid),
    enemyTeam: clean.slice(mid),
    source: "image_upload",
  };
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
    const predictionClasses = parseYoloClasses(parsed);
    return {
      ok: true,
      predictionClasses,
      summary: typeof parsed.summary === "string" ? parsed.summary : "YOLO completed.",
      raw: parsed,
      imageContext: splitTeamClasses(predictionClasses),
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

const HERO_NAMES = [
  "magneto",
  "doctor strange",
  "hulk",
  "venom",
  "captain america",
  "groot",
  "thor",
  "emma frost",
  "the thing",
  "angela",
  "peni parker",
  "deadpool",
  "hela",
  "punisher",
  "hawkeye",
  "namor",
  "winter soldier",
  "spider-man",
  "iron fist",
  "magik",
  "black panther",
  "psylocke",
  "iron man",
  "human torch",
  "storm",
  "scarlet witch",
  "star-lord",
  "moon knight",
  "wolverine",
  "blade",
  "black widow",
  "squirrel girl",
  "phoenix",
  "gambit",
  "rogue",
  "daredevil",
  "elsa bloodstone",
  "mister fantastic",
  "luna snow",
  "mantis",
  "rocket raccoon",
  "loki",
  "jeff the land shark",
  "cloak & dagger",
  "adam warlock",
  "invisible woman",
  "ultron",
  "white fox",
];

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

const MARVEL_DOMAIN_KEYWORDS = [
  "marvel rivals",
  "vanguard",
  "duelist",
  "strategist",
  "team-up",
  "team up",
  "dive",
  "poke",
  "brawl",
  "domination",
  "convergence",
  "convoy",
];

const HERO_ALIASES: Record<string, string> = {
  majik: "magik",
  spiderman: "spider-man",
  starlord: "star-lord",
  jeff: "jeff the land shark",
  cloak: "cloak & dagger",
  dagger: "cloak & dagger",
};

function isRelationshipQuestion(input: string): boolean {
  return /(counter|synerg|works? with|play with|pair|pairing|relation|team[- ]?up|good with|bad against|who fits with)/.test(
    input,
  );
}

function isMarvelRivalsRelated(input: string): boolean {
  if (!input.trim()) {
    return false;
  }
  if (MARVEL_DOMAIN_KEYWORDS.some((keyword) => input.includes(keyword))) {
    return true;
  }
  const normalized = input
    .split(/[^a-z0-9-]+/g)
    .filter(Boolean)
    .map((token) => HERO_ALIASES[token] || token);
  const normalizedInput = normalized.join(" ");

  if (HERO_NAMES.some((hero) => normalizedInput.includes(hero))) {
    return true;
  }
  if (strategyVocabularyCache && normalized.some((token) => strategyVocabularyCache?.has(token))) {
    return true;
  }
  return false;
}

async function ensureStrategyVocabulary(): Promise<void> {
  if (strategyVocabularyCache) {
    return;
  }
  const strategyPath = STRATEGY_PATHS.find((candidate) => existsSync(candidate));
  if (!strategyPath) {
    return;
  }
  const content = await readFile(strategyPath, "utf-8");
  strategyVocabularyCache = new Set(tokenize(content).filter((token) => token.length > 3));
}

function normalizeQueryText(input: string): string {
  const pieces = input
    .toLowerCase()
    .split(/(\s+|[^a-z0-9-]+)/g)
    .filter((part) => part.length > 0);
  return pieces
    .map((part) => {
      const trimmed = part.trim();
      if (!trimmed || /\s+/.test(part) || /[^a-z0-9-]/.test(part)) {
        return part;
      }
      return HERO_ALIASES[trimmed] || trimmed;
    })
    .join("");
}

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
  const relationQuery = isRelationshipQuestion(query.toLowerCase());
  const relationSignal = /(counter|synerg|works? with|pair|team[- ]?up|anti-dive|anti-poke|anti-brawl)/.test(
    chunk.fullText.toLowerCase(),
  );
  const relationBoost = relationQuery && relationSignal ? 2.0 : 0;
  const multiHeroBoost =
    relationQuery && findMentionedHeroes(chunk.fullText).length >= 2 ? 1.2 : 0;
  const density = overlap > 0 ? overlap / Math.max(6, chunk.tokens.length / 14) : 0;
  const score = lexicalScore + headingBoost + phraseBoost + relationBoost + multiHeroBoost + density;

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

function findMentionedHeroes(text: string): string[] {
  const lower = text.toLowerCase();
  return HERO_NAMES.filter((hero) => lower.includes(hero));
}

type Archetype = "dive" | "poke" | "brawl" | "control" | "sustain" | "anti_dive" | "anti_poke" | "anti_brawl";
type RelationType = "works_with" | "counters" | "enables";

type HeroNode = {
  name: string;
  archetypes: Set<Archetype>;
};

type GraphEdge = {
  from: string;
  to: string;
  relation: RelationType;
  reason: string;
};

type KnowledgeGraph = {
  nodes: Map<string, HeroNode>;
  edges: GraphEdge[];
};

function ensureNode(graph: KnowledgeGraph, hero: string): HeroNode {
  const existing = graph.nodes.get(hero);
  if (existing) {
    return existing;
  }
  const created: HeroNode = { name: hero, archetypes: new Set<Archetype>() };
  graph.nodes.set(hero, created);
  return created;
}

function addArchetypesFromText(node: HeroNode, text: string): void {
  const lower = text.toLowerCase();
  if (/(^|[^a-z])dive([^a-z]|$)|diver|dive comp/.test(lower)) node.archetypes.add("dive");
  if (/(^|[^a-z])poke([^a-z]|$)|hitscan|sniper/.test(lower)) node.archetypes.add("poke");
  if (/(^|[^a-z])brawl([^a-z]|$)|rush|deathball/.test(lower)) node.archetypes.add("brawl");
  if (/(^|[^a-z])control([^a-z]|$)|area-control|area control/.test(lower)) node.archetypes.add("control");
  if (/sustain|heal|healing|support/.test(lower)) node.archetypes.add("sustain");
  if (/anti-dive|counter dive|counters dive|anti dive/.test(lower)) node.archetypes.add("anti_dive");
  if (/anti-poke|counter poke|counters poke|anti poke/.test(lower)) node.archetypes.add("anti_poke");
  if (/anti-brawl|counter brawl|counters brawl|anti brawl/.test(lower)) node.archetypes.add("anti_brawl");
}

function pushEdge(graph: KnowledgeGraph, edge: GraphEdge): void {
  const exists = graph.edges.some(
    (candidate) =>
      candidate.from === edge.from &&
      candidate.to === edge.to &&
      candidate.relation === edge.relation &&
      candidate.reason === edge.reason,
  );
  if (!exists) {
    graph.edges.push(edge);
  }
}

function buildKnowledgeGraph(snippets: string[]): KnowledgeGraph {
  const graph: KnowledgeGraph = {
    nodes: new Map<string, HeroNode>(),
    edges: [],
  };

  for (const snippet of snippets) {
    const lower = snippet.toLowerCase();
    const heroes = findMentionedHeroes(snippet);
    if (heroes.length === 0) {
      continue;
    }

    for (const hero of heroes) {
      const node = ensureNode(graph, hero);
      addArchetypesFromText(node, lower);
    }

    // Explicit pair/counter statements from local snippet context.
    if (/(countered by|counters?|hard counter)/.test(lower) && heroes.length >= 2) {
      for (let i = 0; i < heroes.length - 1; i += 1) {
        pushEdge(graph, {
          from: heroes[i],
          to: heroes[i + 1],
          relation: "counters",
          reason: "direct counter statement found in retrieved guide context",
        });
      }
    }

    // Hero chains written like "A + B + C" are usually synergy/core examples.
    if (/\+/.test(lower) && heroes.length >= 2) {
      for (let i = 0; i < heroes.length; i += 1) {
        for (let j = i + 1; j < heroes.length; j += 1) {
          pushEdge(graph, {
            from: heroes[i],
            to: heroes[j],
            relation: "works_with",
            reason: "listed together in a core/composition pattern",
          });
          pushEdge(graph, {
            from: heroes[j],
            to: heroes[i],
            relation: "works_with",
            reason: "listed together in a core/composition pattern",
          });
        }
      }
    }
  }

  // Derived graph edges from archetypes.
  const nodes = Array.from(graph.nodes.values());
  for (const a of nodes) {
    for (const b of nodes) {
      if (a.name === b.name) {
        continue;
      }
      if (a.archetypes.has("anti_dive") && b.archetypes.has("dive")) {
        pushEdge(graph, {
          from: a.name,
          to: b.name,
          relation: "counters",
          reason: "anti-dive relation inferred from archetypes",
        });
      }
      if (a.archetypes.has("anti_poke") && b.archetypes.has("poke")) {
        pushEdge(graph, {
          from: a.name,
          to: b.name,
          relation: "counters",
          reason: "anti-poke relation inferred from archetypes",
        });
      }
      if (a.archetypes.has("anti_brawl") && b.archetypes.has("brawl")) {
        pushEdge(graph, {
          from: a.name,
          to: b.name,
          relation: "counters",
          reason: "anti-brawl relation inferred from archetypes",
        });
      }
      // Positive relation: same archetype heroes generally synergize.
      const synergyArchetypes: Archetype[] = ["dive", "poke", "brawl", "control", "sustain"];
      for (const tag of synergyArchetypes) {
        if (a.archetypes.has(tag) && b.archetypes.has(tag)) {
          pushEdge(graph, {
            from: a.name,
            to: b.name,
            relation: "works_with",
            reason: `both fit ${tag} archetype`,
          });
          break;
        }
      }
      // Enabler relation for sustain supports helping aggressive archetypes.
      if (a.archetypes.has("sustain") && (b.archetypes.has("dive") || b.archetypes.has("brawl"))) {
        pushEdge(graph, {
          from: a.name,
          to: b.name,
          relation: "enables",
          reason: "sustain enables aggressive frontline uptime",
        });
      }
    }
  }

  return graph;
}

function graphToRelationLines(graph: KnowledgeGraph): string[] {
  const lines: string[] = [];
  const topEdges = graph.edges.slice(0, 28);
  for (const edge of topEdges) {
    lines.push(`${edge.from} ${edge.relation.replace("_", " ")} ${edge.to} (${edge.reason}).`);
  }
  return lines;
}

function deriveEntityRelations(snippets: string[]): string[] {
  const graph = buildKnowledgeGraph(snippets);
  const joined = snippets.join("\n\n").toLowerCase();
  const relations = new Set<string>();

  for (const line of graphToRelationLines(graph)) {
    relations.add(line);
  }

  // Archetype-level relation derivations from the guide's matchup triangle.
  if (joined.includes("dive") && joined.includes("poke")) {
    relations.add("Dive generally counters poke when dive can reach backline targets.");
  }
  if (joined.includes("poke") && joined.includes("brawl")) {
    relations.add("Poke generally counters brawl on maps with long sightlines and spacing.");
  }
  if (joined.includes("brawl") && joined.includes("dive")) {
    relations.add("Brawl generally counters dive through grouped sustain and peel.");
  }

  if (joined.includes("namor") && joined.includes("spider-man")) {
    relations.add("Namor is anti-dive and can counter Spider-Man's dive angles.");
  }
  if (joined.includes("black panther") && joined.includes("dive")) {
    relations.add("Black Panther is a dive hero and works best with other dive enablers and dive initiators.");
  }

  return Array.from(relations).slice(0, 30);
}

export async function searchStrategyTool(query: string): Promise<string[]> {
  const strategyPath = STRATEGY_PATHS.find((candidate) => existsSync(candidate));
  if (!strategyPath) {
    return ["No strategy guide file found. Expected marvel_rivals_strategy.md or marvel_rivals_strategy_guide.md."];
  }

  const content = await readFile(strategyPath, "utf-8");
  if (!strategyVocabularyCache) {
    strategyVocabularyCache = new Set(
      tokenize(content).filter((token) => token.length > 3),
    );
  }
  const chunks = splitStrategyChunks(content);

  const normalizedQuery = normalizeQueryText(query);
  const relationQuery = isRelationshipQuestion(normalizedQuery);
  const expandedQuery = relationQuery
    ? `${normalizedQuery} counters synergies works with play with team-up anti-dive anti-poke anti-brawl dive poke brawl`
    : normalizedQuery;
  const queryTokens = tokenize(expandedQuery);
  const mentionedHeroes = findMentionedHeroes(normalizedQuery);
  const ranked = chunks
    .map((chunk) => scoreChunk(chunk, expandedQuery, queryTokens))
    .filter((item) => item.score > 0 || queryTokens.length === 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 12);

  const diverseTop = takeDiverseTop(ranked, 4);

  // Hero-anchored fallback: ensure sections mentioning queried hero aliases are included.
  if (mentionedHeroes.length > 0) {
    const heroAnchored = chunks
      .filter((chunk) => {
        const lower = chunk.fullText.toLowerCase();
        return mentionedHeroes.some((hero) => lower.includes(hero));
      })
      .slice(0, 3)
      .map((chunk) => ({
        chunk,
        overlap: mentionedHeroes.length,
        score: 100, // force include hero-matching context
      }));
    const merged = [...heroAnchored, ...diverseTop];
    const deduped: ScoredChunk[] = [];
    for (const item of merged) {
      if (!deduped.some((d) => d.chunk.fullText === item.chunk.fullText)) {
        deduped.push(item);
      }
      if (deduped.length >= 4) {
        break;
      }
    }
    if (deduped.length > 0) {
      return deduped.map(({ chunk, overlap, score }) =>
        `[${chunk.heading}] score=${score.toFixed(2)} overlap=${overlap}\n${trimSnippet(
          chunk.body,
          700,
        )}`,
      );
    }
  }

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
            "You are a Marvel Rivals strategy agent. You must answer using only facts present in the provided STRATEGY CONTEXT snippets and DERIVED RELATIONS / KNOWLEDGE GRAPH links. Do not use outside knowledge. If the answer is not supported by that context, respond with: 'I don't have enough support in the provided strategy guide context to answer that.' For relationship questions, structure reasoning explicitly as 'A -> relation -> B' and tie each relation to context support.",
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

export async function runAgent(input: {
  userInput: string;
  image: Blob | null;
  layoutPreset: string;
  previousImageContext?: ImageContext | null;
}): Promise<AgentResult> {
  await ensureStrategyVocabulary();
  const intent = detectIntent(input.userInput, Boolean(input.image));
  const tools: ToolUsage[] = [];
  const context: string[] = [];
  const shouldRunStrategy = intent === "strategy_question" || input.userInput.trim().length > 0;
  const shouldRunRelationshipReasoning = intent === "relationship_question";
  const shouldRunExternalApi = intent === "external_api_lookup";

  // Run independent tool calls concurrently to reduce end-to-end latency.
  const yoloTask = input.image ? runYoloTool(input.image, input.layoutPreset) : Promise.resolve(null);
  const strategyTask = shouldRunStrategy ? searchStrategyTool(input.userInput) : Promise.resolve(null);
  const externalApiTask = shouldRunExternalApi
    ? callFutureApiTool(input.userInput)
    : Promise.resolve(null);

  const [yolo, snippets, apiResult] = await Promise.all([yoloTask, strategyTask, externalApiTask]);

  if (yolo) {
    tools.push({
      tool: "runYoloTool",
      ok: yolo.ok,
      details: yolo.ok
        ? `Detected ${yolo.predictionClasses.length} class entries.`
        : yolo.error || "YOLO failed",
    });
    if (yolo.ok) {
      const preview = yolo.predictionClasses.slice(0, 12).join(", ");
      context.push(`YOLO summary: ${yolo.summary}\nDetected classes: ${preview || "(none detected)"}`);
      if (yolo.imageContext) {
        context.push(
          `IMAGE TEAM CONTEXT:\nYour team: ${yolo.imageContext.yourTeam.join(", ") || "(unknown)"}\nEnemy team: ${yolo.imageContext.enemyTeam.join(", ") || "(unknown)"}`,
        );
      }
    }
  }

  if (!input.image && input.previousImageContext) {
    const carry = input.previousImageContext;
    context.push(
      `IMAGE TEAM CONTEXT (from previous upload):\nYour team: ${carry.yourTeam.join(", ") || "(unknown)"}\nEnemy team: ${carry.enemyTeam.join(", ") || "(unknown)"}`,
    );
  }

  if (snippets) {
    const relations = deriveEntityRelations(snippets);
    tools.push({
      tool: "searchStrategyTool",
      ok: snippets.length > 0,
      details: `Retrieved ${snippets.length} strategy snippets.`,
    });
    context.push(`STRATEGY CONTEXT:\n${snippets.map((s, i) => `[${i + 1}] ${s}`).join("\n\n")}`);
    if (relations.length > 0) {
      context.push(`DERIVED RELATIONS:\n${relations.map((r, i) => `[R${i + 1}] ${r}`).join("\n")}`);
    }
    if (shouldRunRelationshipReasoning) {
      context.push(
        "RELATIONSHIP ANSWER FORMAT:\n- Prefer relation triples.\n- Example: Black Panther -> works_with -> Venom (shared dive archetype in context).",
      );
    }
  }

  if (apiResult) {
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
    "Use only STRATEGY CONTEXT and DERIVED RELATIONS for factual claims.",
    "Tool context:",
    context.join("\n\n"),
  ]
    .filter(Boolean)
    .join("\n\n");

  const hasStrategyContext = context.some((entry) => entry.startsWith("STRATEGY CONTEXT:"));
  if (!hasStrategyContext) {
    return {
      intent,
      tools,
      context,
      response: "I don't have enough support in the provided strategy guide context to answer that.",
    };
  }

  const llm = await callOpenAIResponse(prompt);
  return {
    intent,
    tools,
    context,
    response: llm || fallbackResponse(input.userInput, context),
    imageContext: yolo?.imageContext ?? input.previousImageContext ?? undefined,
  };
}
