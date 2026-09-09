import type { ReactNode } from "react";

/**
 * The shared surfaces from DESIGN.md's component list. Every radius, padding and colour here is
 * the design system's, so a page never reaches for a raw hex or a one-off radius.
 */

export function Section({
  id,
  children,
}: {
  id?: string;
  children: ReactNode;
}) {
  return (
    <section id={id} className="pt-20 scroll-mt-24">
      {children}
    </section>
  );
}

export function SectionHead({
  title,
  lede,
}: {
  title: string;
  lede?: string;
}) {
  return (
    <div className="flex flex-col gap-4 mb-10">
      <h2 className="display text-[clamp(40px,5.4vw,64px)] leading-none">{title}</h2>
      {lede ? <p className="max-w-[62ch] text-lg leading-relaxed">{lede}</p> : null}
    </div>
  );
}

/** Limestone surface, 40px radius, no shadow anywhere in this system. */
export function Card({
  children,
  tone = "limestone",
  className = "",
}: {
  children: ReactNode;
  tone?: "limestone" | "ember" | "plasma" | "obsidian";
  className?: string;
}) {
  const tones = {
    limestone: "bg-limestone text-obsidian",
    ember: "bg-ember text-chalk",
    plasma: "bg-plasma text-chalk",
    obsidian: "bg-obsidian text-chalk",
  } as const;
  return (
    <div className={`rounded-card p-card ${tones[tone]} ${className}`}>{children}</div>
  );
}

/** Sulfur pill. DESIGN.md's only yellow element, and only ever a label. */
export function Tag({ children }: { children: ReactNode }) {
  return (
    <span className="inline-block bg-sulfur text-obsidian rounded-pill px-3 py-1 text-xs uppercase tracking-[0.06em]">
      {children}
    </span>
  );
}

export function Caption({ children }: { children: ReactNode }) {
  return (
    <p className="mt-4 max-w-[80ch] text-sm leading-relaxed text-obsidian/70">{children}</p>
  );
}

/** Label and value pair. Used instead of a table when there are only a handful of rows. */
export function KeyValue({ label, value }: { label: string; value: string }) {
  return (
    <div className="row-divider flex justify-between gap-4 py-2.5 text-sm">
      <span>{label}</span>
      <b className="num font-medium text-right">{value}</b>
    </div>
  );
}
