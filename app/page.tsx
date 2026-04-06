"use client";

const sections = [
  {
    title: "PERFORMANCE SUMMARY",
    body: "High Damage/Blocked was notable, but lacked final impact. Focus on burst damage timing to confirm kills.",
  },
  {
    title: "PLAYER FOCUS (Clarks)",
    body: "Prioritize aiming during team fights. Clarks was the primary healing target but had the lowest accuracy at 19%.",
  },
  {
    title: "ENEMY ALERT (Nick xe)",
    body: "Check key ability timings. Enemy Nick xe (25 Eliminations) was not contested efficiently.",
  },
  {
    title: "TEAM STRATEGY",
    body: "Focus on target priority. The opposing team's healing (Stumberjones, awesom1fighter) often significantly out-paced our team. Target their supports earlier.",
  },
];

export default function Home() {
  return (
    <div className="h-screen w-full flex justify-end p-3">
      <aside
        className="w-[300px] h-full rounded-2xl overflow-hidden flex flex-col border border-white/[0.06] shadow-2xl"
        style={{
          background: "rgba(14, 14, 26, 0.88)",
          backdropFilter: "blur(18px)",
          WebkitBackdropFilter: "blur(18px)",
        }}
      >
        <div
          className="flex items-center justify-between px-4 py-3 border-b border-white/[0.06]"
          style={{ WebkitAppRegion: "drag" } as React.CSSProperties}
        >
          <h1 className="text-[11px] font-bold tracking-wide text-blue-400">
            POST-MATCH ANALYSIS &amp; ADVICE
          </h1>
          <button
            onClick={() => (window as any).ipc?.quit()}
            className="w-6 h-6 flex items-center justify-center rounded-md text-zinc-500 hover:text-red-400 hover:bg-red-400/10 transition-colors text-sm font-bold cursor-pointer"
            style={{ WebkitAppRegion: "no-drag" } as React.CSSProperties}
          >
            &times;
          </button>
        </div>

        <div className="flex-1 overflow-y-auto">
          {sections.map((s) => (
            <div
              key={s.title}
              className="px-4 py-3 border-b border-white/[0.04]"
            >
              <h2 className="text-[11px] font-bold tracking-wide text-blue-400 mb-1.5">
                {s.title}
              </h2>
              <p className="text-[12px] leading-relaxed text-zinc-400 font-mono">
                {s.body}
              </p>
            </div>
          ))}
        </div>
      </aside>
    </div>
  );
}
