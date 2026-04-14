type ExternalSource = "web" | "none";

export type ExternalLookupResult = {
  source: ExternalSource;
  snippets: string[];
  detail: string;
};

function pullWebSearchText(data: unknown): string {
  const record = data as {
    output_text?: unknown;
    output?: Array<{
      type?: unknown;
      content?: Array<{
        type?: unknown;
        text?: unknown;
      }>;
    }>;
  };
  if (typeof record.output_text === "string" && record.output_text.trim()) {
    return record.output_text.trim();
  }
  const blocks = record.output || [];
  const out: string[] = [];
  for (const block of blocks) {
    const content = block.content || [];
    for (const item of content) {
      if (item.type === "output_text" && typeof item.text === "string" && item.text.trim()) {
        out.push(item.text.trim());
      }
    }
  }
  return out.join("\n\n").trim();
}

async function fetchOpenAIWebContext(query: string): Promise<ExternalLookupResult> {
  const apiKey = process.env.OPENAI_API_KEY?.trim();
  if (!apiKey) {
    return {
      source: "none",
      snippets: [],
      detail: "OPENAI_API_KEY is required for web-search fallback.",
    };
  }
  try {
    const model = process.env.OPENAI_WEB_MODEL?.trim() || "gpt-4.1-mini";
    const webQuery = `Marvel Rivals strategy lookup: ${query}`;
    const response = await fetch("https://api.openai.com/v1/responses", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model,
        tools: [{ type: "web_search_preview" }],
        input: [
          {
            role: "user",
            content: [
              {
                type: "input_text",
                text:
                  `Search the web for factual Marvel Rivals context for: "${webQuery}". ` +
                  "Return concise bullets with hero archetype/role/counter facts only.",
              },
            ],
          },
        ],
      }),
    });
    if (!response.ok) {
      return {
        source: "web",
        snippets: [],
        detail: `OpenAI web search HTTP ${response.status}`,
      };
    }
    const data = await response.json();
    const text = pullWebSearchText(data);
    if (!text) {
      return {
        source: "web",
        snippets: [],
        detail: "OpenAI web search returned no text.",
      };
    }
    const clipped = text.length > 1600 ? `${text.slice(0, 1600)}...` : text;
    return {
      source: "web",
      snippets: [`[web] ${clipped}`],
      detail: "OpenAI web search fallback returned context.",
    };
  } catch (error) {
    return {
      source: "web",
      snippets: [],
      detail: error instanceof Error ? error.message : "Web lookup failed",
    };
  }
}

export async function externalContextLookup(query: string): Promise<ExternalLookupResult> {
  const web = await fetchOpenAIWebContext(query);
  if (web.snippets.length > 0) {
    return web;
  }
  return web.source === "web" ? web : { source: "none", snippets: [], detail: "No external source configured." };
}
