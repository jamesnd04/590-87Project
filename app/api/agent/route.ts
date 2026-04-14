import { NextRequest, NextResponse } from "next/server";
import { runAgent } from "@/lib/agent/tools";
import type { ImageContext } from "@/lib/agent/tools";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  let form: FormData;
  try {
    form = await req.formData();
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid form data" }, { status: 400 });
  }

  const rawInput = form.get("input");
  const userInput = typeof rawInput === "string" ? rawInput.trim() : "";
  const rawLayout = form.get("layout");
  const layoutPreset = typeof rawLayout === "string" ? rawLayout : "";
  const rawTeamContext = form.get("team_context");
  let previousImageContext: ImageContext | null = null;
  if (typeof rawTeamContext === "string" && rawTeamContext.trim()) {
    try {
      const parsed = JSON.parse(rawTeamContext) as {
        yourTeam?: unknown;
        enemyTeam?: unknown;
      };
      previousImageContext = {
        yourTeam: Array.isArray(parsed.yourTeam)
          ? parsed.yourTeam.filter((x): x is string => typeof x === "string")
          : [],
        enemyTeam: Array.isArray(parsed.enemyTeam)
          ? parsed.enemyTeam.filter((x): x is string => typeof x === "string")
          : [],
        source: "carried_context",
      };
    } catch {
      previousImageContext = null;
    }
  }
  const imageField = form.get("image");
  const image = imageField instanceof Blob ? imageField : null;

  if (!userInput && !image) {
    return NextResponse.json(
      { ok: false, error: "Provide text input and/or an image." },
      { status: 400 },
    );
  }

  try {
    const result = await runAgent({
      userInput,
      image,
      layoutPreset,
      previousImageContext,
    });

    return NextResponse.json({
      ok: true,
      ...result,
    });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        error: error instanceof Error ? error.message : "Agent request failed",
      },
      { status: 500 },
    );
  }
}
