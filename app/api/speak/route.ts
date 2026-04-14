import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  const apiKey = process.env.OPENAI_API_KEY?.trim();
  if (!apiKey) {
    return NextResponse.json({ ok: false, error: "OPENAI_API_KEY is not configured." }, { status: 500 });
  }

  let form: FormData;
  try {
    form = await req.formData();
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid form data." }, { status: 400 });
  }

  const text = form.get("text");
  const input = typeof text === "string" ? text.trim() : "";
  if (!input) {
    return NextResponse.json({ ok: false, error: "Missing text input for speech." }, { status: 400 });
  }

  const model = process.env.OPENAI_TTS_MODEL?.trim() || "gpt-4o-mini-tts";
  const voice = process.env.OPENAI_TTS_VOICE?.trim() || "alloy";

  const upstream = await fetch("https://api.openai.com/v1/audio/speech", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      voice,
      input,
      format: "mp3",
    }),
  });

  if (!upstream.ok) {
    const err = (await upstream.json()) as { error?: { message?: string } };
    return NextResponse.json(
      { ok: false, error: err.error?.message || "OpenAI speech generation failed." },
      { status: 500 },
    );
  }

  const audio = await upstream.arrayBuffer();
  return new NextResponse(audio, {
    status: 200,
    headers: {
      "Content-Type": "audio/mpeg",
      "Cache-Control": "no-store",
    },
  });
}
