import { mkdir, readFile, unlink, writeFile } from "fs/promises";
import path from "path";
import { externalContextLookup } from "@/lib/agent/external";
import { semanticRetrieve } from "@/lib/agent/semantic";
import { guessExt, runImageInfer } from "@/lib/inference-utils";

const STRATEGY_PATHS = [
  path.join(process.cwd(), "marvel_rivals_strategy.md"),
  path.join(process.cwd(), "marvel_rivals_strategy_guide.md"),
];
let strategyVocabularyCache: Set<string> | null = null;

type Intent =
  | "image_analysis"
  | "strategy_question"
  | "hero_data_lookup"
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
  if (isCompositionOrMapQuestion(normalized)) {
    return "strategy_question";
  }
  if (isHeroDataLookupQuestion(normalized)) {
    return "hero_data_lookup";
  }
  if (/(api|endpoint|external|live data|fetch data|stats)/.test(normalized)) {
    return "external_api_lookup";
  }
  if (/(how|what|why|guide|strategy|pick|counter|team|hero|comp|who|which|list)/.test(normalized)) {
    return "strategy_question";
  }
  return "general_chat";
}

function isHeroDataLookupQuestion(input: string): boolean {
  return /(ability|abilities|cooldown|hero info|hero details|skin|skins|achievement|achievements|item|items|nameplate|mvp|emote|spray|player profile|match history|search player|username)/.test(
    input,
  );
}

function isCompositionOrMapQuestion(input: string): boolean {
  return /(composition|comp|team comp|map|maps|domination|convoy|convergence|lineup|draft)/.test(input);
}

function isGeneralHeroQuestion(input: string): boolean {
  const normalized = normalizeQueryText(input);
  const heroes = findMentionedHeroes(normalized);
  if (heroes.length === 0) {
    return false;
  }
  return !isRelationshipQuestion(normalized) && !isCompositionOrMapQuestion(normalized);
}

function buildSearchPlan(intent: Intent, userInput: string): SearchSource[] {
  const normalized = userInput.toLowerCase();
  const plan: SearchSource[] = [];

  if (intent === "hero_data_lookup" || isGeneralHeroQuestion(normalized)) {
    plan.push("hero_ability_tool", "external_web", "strategy_rag");
  } else if (intent === "relationship_question" || isCompositionOrMapQuestion(normalized)) {
    plan.push("strategy_rag", "hero_ability_tool", "external_web");
  } else if (intent === "external_api_lookup") {
    plan.push("external_web", "strategy_rag", "hero_ability_tool");
  } else {
    plan.push("strategy_rag", "hero_ability_tool", "external_web");
  }

  return Array.from(new Set(plan));
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

/** Shown to the user immediately before the main answer when an image was analyzed. */
function formatCharactersDetectedPrefix(yolo: YoloResult | null): string {
  if (!yolo) {
    return "Characters detected from image:\n(no detection result)";
  }
  if (!yolo.ok) {
    return `Characters detected from image:\n(detection failed: ${yolo.error || "unknown error"})`;
  }
  const ic = yolo.imageContext;
  const classes = yolo.predictionClasses;
  const lines: string[] = ["Characters detected from image:"];
  if (ic && (ic.yourTeam.length > 0 || ic.enemyTeam.length > 0)) {
    lines.push(`Your team: ${ic.yourTeam.join(", ") || "(none)"}`);
    lines.push(`Enemy team: ${ic.enemyTeam.join(", ") || "(none)"}`);
  } else if (classes.length > 0) {
    lines.push(classes.join(", "));
  } else {
    lines.push("(no character classes returned)");
  }
  return lines.join("\n");
}

function prependDetectionToResponse(image: Blob | null, yolo: YoloResult | null, body: string): string {
  if (!image) {
    return body;
  }
  const prefix = formatCharactersDetectedPrefix(yolo);
  return `${prefix}\n\n${body}`;
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

type HeroGuideRow = {
  hero: string;
  role: string;
  styleTags: string;
  summary: string;
  roleGroup: "vanguard" | "duelist" | "strategist" | "unknown";
};

type HeroFacetQuery = {
  styleTag: string | null;
  roleTag: string | null;
};

type ScoredChunk = {
  id: string;
  chunk: StrategyChunk;
  score: number;
  overlap: number;
};

type RetrievalSource = "local_guide" | "mcp" | "web";

type StrategyRetrievalDiagnostics = {
  normalizedQuery: string;
  expandedQuery: string;
  heroes: string[];
  lexicalTopScore: number;
  semanticTopScore: number;
  confidence: number;
  source: RetrievalSource;
  fallbackReason?: string;
};

type StrategySearchResult = {
  snippets: string[];
  diagnostics: StrategyRetrievalDiagnostics;
};

type HeroAbilityToolName =
  | "listHeroes"
  | "getHeroAbilities"
  | "getHeroInfo"
  | "getHeroSkins"
  | "listSkins"
  | "listAchievements"
  | "searchAchievement"
  | "listItems"
  | "getItemsByType"
  | "listMaps"
  | "filterMaps"
  | "getPlayerProfile"
  | "searchPlayer"
  | "getPlayerMatchHistory";

type HeroAbilityToolIntent = {
  tool: HeroAbilityToolName;
  subject?: string;
};

type SearchSource = "strategy_rag" | "hero_ability_tool" | "external_web";

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

function scoreChunk(chunk: StrategyChunk, id: string, query: string, queryTokens: string[]): ScoredChunk {
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
    id,
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

function parseMentionedItemType(query: string): "NAMEPLATE" | "MVP" | "EMOTE" | "SPRAY" | null {
  const lower = query.toLowerCase();
  if (lower.includes("nameplate")) return "NAMEPLATE";
  if (/\bmvp\b/.test(lower)) return "MVP";
  if (lower.includes("emote")) return "EMOTE";
  if (lower.includes("spray")) return "SPRAY";
  return null;
}

function resolveHeroAbilityToolIntent(query: string): HeroAbilityToolIntent {
  const lower = normalizeQueryText(query);
  const heroes = findMentionedHeroes(lower);
  const hero = heroes[0];
  const itemType = parseMentionedItemType(lower);

  if (/(match history|recent matches|last matches)/.test(lower)) {
    return { tool: "getPlayerMatchHistory", subject: query };
  }
  if (/(player profile|profile stats)/.test(lower)) {
    return { tool: "getPlayerProfile", subject: query };
  }
  if (/(search player|find player|username)/.test(lower)) {
    return { tool: "searchPlayer", subject: query };
  }
  if (/(filter maps|map type|convoy|domination|convergence)/.test(lower)) {
    return { tool: "filterMaps", subject: query };
  }
  if (/\bmaps?\b/.test(lower)) {
    return { tool: "listMaps" };
  }
  if (/(search achievement|achievement .*named|achievement .*called)/.test(lower)) {
    return { tool: "searchAchievement", subject: query };
  }
  if (/\bachievements?\b/.test(lower)) {
    return { tool: "listAchievements" };
  }
  if (itemType) {
    return { tool: "getItemsByType", subject: itemType };
  }
  if (/\bitems?\b/.test(lower)) {
    return { tool: "listItems" };
  }
  if (/\bskins?\b/.test(lower) && hero) {
    return { tool: "getHeroSkins", subject: hero };
  }
  if (/\bskins?\b/.test(lower)) {
    return { tool: "listSkins" };
  }
  if (/(abilities|ability|cooldown|kit|skills)/.test(lower) && hero) {
    return { tool: "getHeroAbilities", subject: hero };
  }
  if ((/(hero info|hero details|tell me about|what does .* do)/.test(lower) || hero) && hero) {
    return { tool: "getHeroInfo", subject: hero };
  }
  return { tool: "listHeroes" };
}

function semanticRagEnabled(): boolean {
  return process.env.ENABLE_SEMANTIC_RAG === "1";
}

function extractHeroGuideRows(content: string): HeroGuideRow[] {
  const lines = content.split("\n");
  const rows: HeroGuideRow[] = [];
  let currentRoleGroup: HeroGuideRow["roleGroup"] = "unknown";
  let inHeroPlaystyleIndex = false;
  for (const line of lines) {
    const heading = line.trim().toLowerCase();
    if (heading === "## 9. hero playstyle index") {
      inHeroPlaystyleIndex = true;
      currentRoleGroup = "unknown";
      continue;
    }
    if (!inHeroPlaystyleIndex) {
      continue;
    }
    if (/^##\s+/.test(heading) && heading !== "## 9. hero playstyle index") {
      break;
    }
    if (heading === "### vanguards") {
      currentRoleGroup = "vanguard";
      continue;
    }
    if (heading === "### duelists") {
      currentRoleGroup = "duelist";
      continue;
    }
    if (heading === "### strategists") {
      currentRoleGroup = "strategist";
      continue;
    }
    if (!line.startsWith("|")) {
      continue;
    }
    const cells = line
      .split("|")
      .map((part) => part.trim())
      .filter(Boolean);
    if (cells.length < 3) {
      continue;
    }
    const heroCell = cells[0] || "";
    const styleCell = cells[1] || "";
    const summaryCell = cells[2] || "";
    const roleCell = currentRoleGroup === "unknown" ? "" : currentRoleGroup;
    const hero = heroCell.replace(/\*\*/g, "").trim().toLowerCase();
    if (!hero || hero === "hero") {
      continue;
    }
    rows.push({
      hero,
      role: roleCell.replace(/\*\*/g, "").trim(),
      styleTags: styleCell.replace(/\*\*/g, "").trim(),
      summary: summaryCell.replace(/\*\*/g, "").trim(),
      roleGroup: currentRoleGroup,
    });
  }
  return rows;
}

function expandHeroAliases(hero: string): string[] {
  const aliases = [hero];
  const compactHero = hero.replace(/[^a-z0-9]/g, "");
  aliases.push(compactHero);
  for (const [alias, canonical] of Object.entries(HERO_ALIASES)) {
    if (canonical === hero) {
      aliases.push(alias);
    }
  }
  return Array.from(new Set(aliases));
}

function rewriteQueryForRetrieval(normalizedQuery: string, heroes: string[]): string {
  const lower = normalizedQuery.toLowerCase();
  const asksPlaystyle = /(style|playstyle|archetype|role|what is|what style)/.test(lower);
  const heroHints = heroes.length > 0 ? `${heroes.join(" ")} hero` : "";
  if (asksPlaystyle) {
    return `${normalizedQuery} ${heroHints} playstyle archetype role dive poke brawl`.trim();
  }
  return normalizedQuery;
}

function parseHeroFacetQuery(normalizedQuery: string): HeroFacetQuery | null {
  const q = normalizedQuery.toLowerCase();
  const isListStyle = /(who are|which heroes|list|show|what are)/.test(q);
  if (!isListStyle) {
    return null;
  }

  const styleTag = /\bdive\b/.test(q)
    ? "dive"
    : /\bpoke\b/.test(q)
      ? "poke"
      : /\bbrawl\b/.test(q)
        ? "brawl"
        : /\bcontrol\b/.test(q)
          ? "control"
          : /\bsustain\b/.test(q)
            ? "sustain"
            : null;

  const roleTag = /\bduelist(s)?\b/.test(q)
    ? "duelist"
    : /\bvanguard(s)?\b/.test(q)
      ? "vanguard"
      : /\bstrategist(s)?\b/.test(q)
        ? "strategist"
        : null;

  if (!styleTag && !roleTag) {
    return null;
  }
  return { styleTag, roleTag };
}

function inferRoleFromHeroRow(row: HeroGuideRow): string | null {
  if (row.roleGroup !== "unknown") {
    return row.roleGroup;
  }
  const text = `${row.role} ${row.styleTags} ${row.summary}`.toLowerCase();
  if (/\bduelist\b/.test(text) || /(melee assassin|assassin|marksman)/.test(text)) {
    return "duelist";
  }
  if (/\bvanguard\b/.test(text) || /(tank|frontline|initiator|bruiser)/.test(text)) {
    return "vanguard";
  }
  if (/\bstrategist\b/.test(text) || /(support|heal|healer|sustain)/.test(text)) {
    return "strategist";
  }
  return null;
}

function parseStyleTagTokens(styleTags: string): string[] {
  return styleTags
    .toLowerCase()
    .split(/[\/,|]+/g)
    .map((part) => part.trim().replace(/\s+/g, "-"))
    .filter(Boolean);
}

function runDeterministicFacetLookup(normalizedQuery: string, heroRows: HeroGuideRow[]): string[] | null {
  const facet = parseHeroFacetQuery(normalizedQuery);
  if (!facet) {
    return null;
  }

  const matched = heroRows.filter((row) => {
    const styleTokens = parseStyleTagTokens(row.styleTags);
    const roleBag = `${row.roleGroup} ${row.role}`.toLowerCase();
    const styleOk = facet.styleTag ? styleTokens.includes(facet.styleTag) : true;
    const inferredRole = inferRoleFromHeroRow(row);
    const roleOk = facet.roleTag ? inferredRole === facet.roleTag : true;
    return styleOk && roleOk && roleBag.length > 0;
  });

  if (matched.length === 0) {
    return [];
  }

  const heroes = matched.map((row) => row.hero).sort((a, b) => a.localeCompare(b));
  const title = [facet.styleTag, facet.roleTag].filter(Boolean).join(" ");
  const header = `[Facet match] ${title || "heroes"} (${heroes.length})\n${heroes.join(", ")}`;
  const evidence = matched
    .slice(0, 5)
    .map((row) => `[Hero row] ${row.hero} | role=${row.roleGroup}/${row.role} | style=${row.styleTags}\n${row.summary}`);
  return [header, ...evidence];
}

function retrievalConfidence(lexicalTopScore: number, semanticTopScore: number, heroes: string[]): number {
  const heroBoost = heroes.length > 0 ? 0.1 : 0;
  return Math.max(0, Math.min(1, lexicalTopScore / 8 + semanticTopScore * 0.6 + heroBoost));
}

function formatScoredSnippet(item: ScoredChunk): string {
  return `[${item.chunk.heading}] score=${item.score.toFixed(2)} overlap=${item.overlap}\n${trimSnippet(item.chunk.body, 700)}`;
}

export async function searchStrategyTool(query: string): Promise<StrategySearchResult> {
  const strategyPath = STRATEGY_PATHS.find((candidate) => existsSync(candidate));
  if (!strategyPath) {
    return {
      snippets: ["No strategy guide file found. Expected marvel_rivals_strategy.md or marvel_rivals_strategy_guide.md."],
      diagnostics: {
        normalizedQuery: query,
        expandedQuery: query,
        heroes: [],
        lexicalTopScore: 0,
        semanticTopScore: 0,
        confidence: 0,
        source: "local_guide",
        fallbackReason: "missing_strategy_file",
      },
    };
  }

  const content = await readFile(strategyPath, "utf-8");
  if (!strategyVocabularyCache) {
    strategyVocabularyCache = new Set(
      tokenize(content).filter((token) => token.length > 3),
    );
  }
  const chunks = splitStrategyChunks(content);
  const heroRows = extractHeroGuideRows(content);

  const normalizedQuery = normalizeQueryText(query);
  const facetSnippets = runDeterministicFacetLookup(normalizedQuery, heroRows);
  if (facetSnippets && facetSnippets.length > 0) {
    return {
      snippets: facetSnippets,
      diagnostics: {
        normalizedQuery,
        expandedQuery: normalizedQuery,
        heroes: findMentionedHeroes(normalizedQuery),
        lexicalTopScore: 1,
        semanticTopScore: 1,
        confidence: 1,
        source: "local_guide",
        fallbackReason: "deterministic_facet_lookup",
      },
    };
  }
  const mentionedHeroes = findMentionedHeroes(normalizedQuery);
  const expandedQuery = rewriteQueryForRetrieval(normalizedQuery, mentionedHeroes);
  const relationQuery = isRelationshipQuestion(normalizedQuery);
  const retrievalQuery = relationQuery
    ? `${expandedQuery} counters synergies works with play with team-up anti-dive anti-poke anti-brawl dive poke brawl`
    : expandedQuery;
  const queryTokens = tokenize(retrievalQuery);
  const ranked = chunks
    .map((chunk, idx) => scoreChunk(chunk, `chunk-${idx}`, retrievalQuery, queryTokens))
    .filter((item) => item.score > 0 || queryTokens.length === 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 12);
  const lexicalTopScore = ranked[0]?.score ?? 0;

  const semanticDocs = [
    ...chunks.map((chunk, idx) => ({ id: `chunk-${idx}`, text: chunk.fullText })),
    ...heroRows.map((row, idx) => ({
      id: `hero-row-${idx}`,
      text: `${row.hero} ${row.roleGroup} ${row.role} ${row.styleTags} ${row.summary}`,
    })),
  ];
  const semantic = await semanticRetrieve({
    enabled: semanticRagEnabled(),
    query: retrievalQuery,
    docs: semanticDocs,
    topK: 8,
    corpusId: `${strategyPath}:${content.length}`,
  });
  const semanticTopScore = semantic[0]?.score ?? 0;

  const mergedScores = new Map<string, ScoredChunk>();
  for (const item of ranked) {
    mergedScores.set(item.id, item);
  }
  for (const item of semantic) {
    if (!item.id.startsWith("chunk-")) {
      continue;
    }
    const existing = mergedScores.get(item.id);
    if (existing) {
      existing.score += item.score * 4;
      continue;
    }
    const idx = Number(item.id.replace("chunk-", ""));
    const chunk = chunks[idx];
    if (!chunk) {
      continue;
    }
    mergedScores.set(item.id, {
      id: item.id,
      chunk,
      overlap: findMentionedHeroes(chunk.fullText).length,
      score: item.score * 4,
    });
  }
  const mergedRanked = Array.from(mergedScores.values()).sort((a, b) => b.score - a.score).slice(0, 16);

  const diverseTop = takeDiverseTop(mergedRanked, 4);
  const confidence = retrievalConfidence(lexicalTopScore, semanticTopScore, mentionedHeroes);
  let fallbackReason: string | undefined;

  // Hero-anchored fallback: ensure sections mentioning queried hero aliases are included.
  if (mentionedHeroes.length > 0) {
    const aliasSet = new Set<string>();
    for (const hero of mentionedHeroes) {
      for (const alias of expandHeroAliases(hero)) {
        aliasSet.add(alias);
      }
    }
    const heroAnchored = chunks
      .filter((chunk) => {
        const lower = chunk.fullText.toLowerCase();
        for (const alias of aliasSet) {
          if (lower.includes(alias)) {
            return true;
          }
        }
        return false;
      })
      .slice(0, 3)
      .map((chunk) => ({
        id: `forced-${chunk.heading}`,
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
      // Pull in direct hero table rows for compact "what style is X" prompts.
      const heroRowsMatched = heroRows
        .filter((row) => mentionedHeroes.includes(row.hero))
        .slice(0, 2)
        .map((row) => `[Hero index] ${row.hero} | role=${row.role} | style=${row.styleTags}\n${row.summary}`);
      return {
        snippets: [...heroRowsMatched, ...deduped.map(formatScoredSnippet)].slice(0, 5),
        diagnostics: {
          normalizedQuery,
          expandedQuery: retrievalQuery,
          heroes: mentionedHeroes,
          lexicalTopScore,
          semanticTopScore,
          confidence,
          source: "local_guide",
        },
      };
    }
    fallbackReason = "hero_mentioned_but_no_anchor";
  }

  if (diverseTop.length > 0) {
    return {
      snippets: diverseTop.map(formatScoredSnippet),
      diagnostics: {
        normalizedQuery,
        expandedQuery: retrievalQuery,
        heroes: mentionedHeroes,
        lexicalTopScore,
        semanticTopScore,
        confidence,
        source: "local_guide",
      },
    };
  }

  return {
    snippets: [],
    diagnostics: {
      normalizedQuery,
      expandedQuery: retrievalQuery,
      heroes: mentionedHeroes,
      lexicalTopScore,
      semanticTopScore,
      confidence,
      source: "local_guide",
      fallbackReason: fallbackReason || "lexical_empty",
    },
  };
}

export async function callFutureApiTool(_query: string): Promise<{ ok: boolean; result: string }> {
  // Placeholder for the upcoming external API integration.
  return {
    ok: false,
    result: "TODO: external API tool not implemented yet.",
  };
}

async function callHeroAbilityTool(query: string): Promise<{ ok: boolean; tool: HeroAbilityToolName; result: string }> {
  const intent = resolveHeroAbilityToolIntent(query);
  const scopedQuery = intent.subject
    ? `${intent.tool} for "${intent.subject}": ${query}`
    : `${intent.tool}: ${query}`;
  const external = await externalContextLookup(scopedQuery);
  if (external.snippets.length === 0) {
    return {
      ok: false,
      tool: intent.tool,
      result: `No ${intent.tool} context returned (${external.detail}).`,
    };
  }
  return {
    ok: true,
    tool: intent.tool,
    result: `[SOURCE: ${external.source}] ${external.snippets.join("\n")}`,
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
            "You are a Marvel Rivals strategy agent. Treat provided tool context as your only factual source. Tool policy: hero-general and hero-ability questions should prefer HERO ABILITY TOOL CONTEXT or EXTERNAL SOURCE context; composition/map/relationship questions should prefer STRATEGY CONTEXT and DERIVED RELATIONS. If one source is empty, continue with other provided sources. If all sources are empty or unsupported, respond exactly: 'I don't have enough support in the available context sources to answer that.' For relationship questions, structure reasoning as 'A -> relation -> B' and tie each relation to context evidence.",
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
  const shouldRunStrategy = input.userInput.trim().length > 0;
  const shouldRunRelationshipReasoning = intent === "relationship_question";
  const shouldRunExternalApi = intent === "external_api_lookup";
  const searchPlan = buildSearchPlan(intent, input.userInput);

  // Run image analysis in parallel; text source retrieval follows a plan with fallback chaining.
  const yoloTask = input.image ? runYoloTool(input.image, input.layoutPreset) : Promise.resolve(null);
  const yolo = await yoloTask;
  let snippets: string[] = [];
  let retrievalDiagnostics: StrategyRetrievalDiagnostics | null = null;
  let heroAbilityResult: { ok: boolean; tool: HeroAbilityToolName; result: string } | null = null;
  let externalResult: { source: string; snippets: string[]; detail: string } | null = null;

  for (const source of searchPlan) {
    if (source === "strategy_rag" && shouldRunStrategy) {
      const strategyResult = await searchStrategyTool(input.userInput);
      retrievalDiagnostics = strategyResult.diagnostics;
      tools.push({
        tool: "searchStrategyTool",
        ok: strategyResult.snippets.length > 0,
        details: `Retrieved ${strategyResult.snippets.length} strategy snippets. confidence=${strategyResult.diagnostics.confidence.toFixed(2)}`,
      });
      if (strategyResult.snippets.length > 0) {
        snippets = strategyResult.snippets;
        break;
      }
      continue;
    }

    if (source === "hero_ability_tool") {
      const result = await callHeroAbilityTool(input.userInput);
      heroAbilityResult = result;
      tools.push({
        tool: result.tool,
        ok: result.ok,
        details: result.result,
      });
      if (result.ok) {
        break;
      }
      continue;
    }

    if (source === "external_web") {
      const ext = await externalContextLookup(input.userInput);
      externalResult = ext;
      tools.push({
        tool: "externalContextLookup",
        ok: ext.snippets.length > 0,
        details: `[SOURCE: ${ext.source}] ${ext.detail}`,
      });
      if (ext.snippets.length > 0) {
        snippets = [...snippets, ...ext.snippets].slice(0, 6);
        break;
      }
    }
  }

  const apiResult = shouldRunExternalApi ? await callFutureApiTool(input.userInput) : null;

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

  if (retrievalDiagnostics) {
    context.push(
      `RETRIEVAL DIAGNOSTICS:\nquery=${retrievalDiagnostics.normalizedQuery}\nexpanded=${retrievalDiagnostics.expandedQuery}\nheroes=${retrievalDiagnostics.heroes.join(", ") || "(none)"}\nlexical_top=${retrievalDiagnostics.lexicalTopScore.toFixed(3)}\nsemantic_top=${retrievalDiagnostics.semanticTopScore.toFixed(3)}\nconfidence=${retrievalDiagnostics.confidence.toFixed(3)}\nsource=${retrievalDiagnostics.source}${retrievalDiagnostics.fallbackReason ? `\nfallback_reason=${retrievalDiagnostics.fallbackReason}` : ""}`,
    );
  }

  if (externalResult) {
    context.push(
      `EXTERNAL SOURCE: ${externalResult.source}\n${externalResult.detail}\n${externalResult.snippets.join("\n")}`,
    );
  }

  if (snippets.length > 0) {
    const relations = deriveEntityRelations(snippets);
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

  if (heroAbilityResult?.ok) {
    context.push(`HERO ABILITY TOOL CONTEXT (${heroAbilityResult.tool}):\n${heroAbilityResult.result}`);
  }

  const prompt = [
    `User intent: ${intent}`,
    `User input: ${input.userInput || "(empty)"}`,
    "Use only provided tool context (STRATEGY CONTEXT, DERIVED RELATIONS, HERO ABILITY TOOL CONTEXT, EXTERNAL SOURCE) for factual claims.",
    "Tool context:",
    context.join("\n\n"),
  ]
    .filter(Boolean)
    .join("\n\n");

  const hasKnowledgeContext =
    context.some((entry) => entry.startsWith("STRATEGY CONTEXT:")) ||
    context.some((entry) => entry.startsWith("HERO ABILITY TOOL CONTEXT")) ||
    context.some((entry) => entry.startsWith("EXTERNAL SOURCE:"));
  if (!hasKnowledgeContext) {
    const unsupported =
      "I don't have enough support in the available context sources to answer that.";
    return {
      intent,
      tools,
      context,
      response: prependDetectionToResponse(input.image, yolo, unsupported),
      imageContext: yolo?.imageContext ?? input.previousImageContext ?? undefined,
    };
  }

  const llm = await callOpenAIResponse(prompt);
  const mainBody = llm || fallbackResponse(input.userInput, context);
  return {
    intent,
    tools,
    context,
    response: prependDetectionToResponse(input.image, yolo, mainBody),
    imageContext: yolo?.imageContext ?? input.previousImageContext ?? undefined,
  };
}
