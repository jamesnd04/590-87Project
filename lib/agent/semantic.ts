import { existsSync } from "fs";
import { mkdir, readFile, writeFile } from "fs/promises";
import path from "path";

type EmbeddingCacheFile = {
  corpusId: string;
  model: string;
  vectors: Record<string, number[]>;
};

type SemanticDoc = {
  id: string;
  text: string;
};

type SemanticResult = {
  id: string;
  score: number;
};

const CACHE_DIR = path.join(process.cwd(), ".cache");
const CACHE_FILE = path.join(CACHE_DIR, "agent-embeddings.json");

async function readEmbeddingCache(): Promise<EmbeddingCacheFile | null> {
  if (!existsSync(CACHE_FILE)) {
    return null;
  }
  try {
    const raw = await readFile(CACHE_FILE, "utf-8");
    const parsed = JSON.parse(raw) as EmbeddingCacheFile;
    if (!parsed || typeof parsed !== "object") {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

async function writeEmbeddingCache(cache: EmbeddingCacheFile): Promise<void> {
  await mkdir(CACHE_DIR, { recursive: true });
  await writeFile(CACHE_FILE, JSON.stringify(cache), "utf-8");
}

async function embedTexts(model: string, texts: string[], apiKey: string): Promise<number[][]> {
  if (texts.length === 0) {
    return [];
  }
  const response = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      input: texts,
    }),
  });
  if (!response.ok) {
    return [];
  }
  const data = (await response.json()) as { data?: Array<{ embedding?: number[] }> };
  return (
    data.data
      ?.map((row) => (Array.isArray(row.embedding) ? row.embedding : []))
      .filter((row) => row.length > 0) || []
  );
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) {
    return 0;
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) {
    return 0;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export async function semanticRetrieve(params: {
  enabled: boolean;
  query: string;
  docs: SemanticDoc[];
  topK: number;
  corpusId: string;
}): Promise<SemanticResult[]> {
  if (!params.enabled || params.docs.length === 0) {
    return [];
  }
  const apiKey = process.env.OPENAI_API_KEY?.trim();
  if (!apiKey) {
    return [];
  }
  const model = process.env.OPENAI_EMBED_MODEL?.trim() || "text-embedding-3-small";
  const ids = params.docs.map((d) => d.id);
  const cache = await readEmbeddingCache();
  const vectors: Record<string, number[]> =
    cache && cache.corpusId === params.corpusId && cache.model === model ? { ...cache.vectors } : {};

  const missingDocs = params.docs.filter((doc) => !vectors[doc.id]);
  if (missingDocs.length > 0) {
    const embedded = await embedTexts(
      model,
      missingDocs.map((doc) => doc.text.slice(0, 4000)),
      apiKey,
    );
    for (let i = 0; i < missingDocs.length && i < embedded.length; i += 1) {
      vectors[missingDocs[i].id] = embedded[i];
    }
    await writeEmbeddingCache({
      corpusId: params.corpusId,
      model,
      vectors,
    });
  }

  const queryEmb = await embedTexts(model, [params.query], apiKey);
  const q = queryEmb[0];
  if (!q) {
    return [];
  }

  return ids
    .map((id) => ({
      id,
      score: cosineSimilarity(q, vectors[id] || []),
    }))
    .filter((item) => Number.isFinite(item.score) && item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, params.topK);
}
