import { NextRequest, NextResponse } from "next/server";
import { runAgent } from "@/lib/agent/tools";

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
