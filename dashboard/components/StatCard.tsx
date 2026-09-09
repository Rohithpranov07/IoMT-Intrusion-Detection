import type { Stat } from "@/lib/data";

/**
 * DESIGN.md's Stat Feature Card: the system's most visually dominant element after the hero.
 * `quiet` is the outlined variant, used where a figure is context rather than a headline.
 */
export function StatCard({ stat }: { stat: Stat }) {
  const tones = {
    ember: "bg-ember text-chalk border-transparent",
    plasma: "bg-plasma text-chalk border-transparent",
    quiet: "bg-limestone text-obsidian border-obsidian",
  } as const;

  return (
    <div
      className={`flex flex-col gap-2 rounded-card p-card border-[1.5px] ${
        tones[stat.tone ?? "ember"]
      }`}
    >
      <span className="text-xs uppercase tracking-[0.08em]">{stat.label}</span>
      <span className="display num text-5xl md:text-6xl leading-[1.05]">{stat.value}</span>
      <span className="text-sm leading-snug">{stat.note}</span>
    </div>
  );
}
