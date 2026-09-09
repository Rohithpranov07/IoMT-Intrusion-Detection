"use client";

import { useState } from "react";
import {
  ATTACK_SCALES,
  CROSS_PRODUCT,
  LATENCIES,
  type RuleKey,
} from "@/lib/data";

const RULES: RuleKey[] = ["fixed", "ewma", "criticality"];

/**
 * The 48-run cross product, one rule at a time.
 *
 * This is the one place on the page where interaction earns itself: a static slice through this
 * cube misattributes the ewma rule's failure. The `atk=60` row alone reads "ewma fails above
 * 500 us"; the `lat=500` column alone reads "ewma is fine at every attack scale". Both are wrong,
 * and only the whole grid shows why.
 *
 * State is encoded by fill AND by the printed figure, never by colour alone.
 */
export function CrossProduct() {
  const [rule, setRule] = useState<RuleKey>("ewma");
  const data = CROSS_PRODUCT[rule];

  return (
    <div>
      <div className="mb-6 flex flex-wrap gap-2" role="group" aria-label="Select threshold rule">
        {RULES.map((key) => (
          <button
            key={key}
            type="button"
            onClick={() => setRule(key)}
            aria-pressed={rule === key}
            className="cursor-pointer rounded-pill border-[1.5px] border-obsidian px-4 py-3 text-sm sm:px-5 sm:py-2.5 transition-colors aria-pressed:border-ember aria-pressed:bg-ember aria-pressed:text-chalk"
          >
            {key}
          </button>
        ))}
      </div>

      <div className="overflow-x-auto">
        <div
          className="grid min-w-[560px] gap-2"
          style={{ gridTemplateColumns: "auto repeat(4, 1fr)" }}
          aria-live="polite"
        >
          <div />
          {ATTACK_SCALES.map((scale) => (
            <div key={scale} className="text-center text-xs uppercase tracking-[0.06em]">
              {scale}
            </div>
          ))}

          {LATENCIES.map((latency, row) => (
            <Row
              key={latency}
              latency={latency}
              blocked={data.blocked[row]}
              falseAlerts={data.falseAlerts[row]}
            />
          ))}
        </div>
      </div>

      <p className="mt-4 text-xs uppercase tracking-[0.08em]">
        Large figure: attacker-bots blocked. Small figure: false-positive alerts raised against
        legitimate wearables.
      </p>
    </div>
  );
}

function Row({
  latency,
  blocked,
  falseAlerts,
}: {
  latency: string;
  blocked: number[];
  falseAlerts: number[];
}) {
  return (
    <>
      <div className="self-center pr-2 text-xs uppercase tracking-[0.06em]">{latency}</div>
      {blocked.map((value, index) => (
        <div
          key={index}
          className={`flex min-h-[92px] flex-col items-center justify-center gap-1 rounded-small border-[1.5px] p-3 text-center ${
            value > 0
              ? "border-transparent bg-ember text-chalk"
              : "border-dotted border-obsidian bg-limestone text-obsidian"
          }`}
        >
          <span className="display num text-3xl leading-none">{value}%</span>
          <span className="num text-xs leading-tight">{falseAlerts[index]} false alerts</span>
        </div>
      ))}
    </>
  );
}
