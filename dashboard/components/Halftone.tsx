"use client";

import { useEffect, useRef } from "react";

/**
 * DESIGN.md's signature motif: orange halftone dots over a violet ground, density rising toward
 * the top right until the field reads as solid ember.
 *
 * Canvas rather than authored SVG path data, because this is a generative field of roughly 1,800
 * dots. It is a client leaf so the rest of the page stays a server component.
 */
export function Halftone() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    const { width, height } = canvas;
    ctx.fillStyle = "#524ae9";
    ctx.fillRect(0, 0, width, height);

    const step = 13;
    const maxRadius = step * 0.46;

    for (let y = 0; y < height; y += step) {
      for (let x = 0; x < width; x += step) {
        const t = (x / width) * 0.62 + (1 - y / height) * 0.38;
        const radius = maxRadius * Math.min(1, Math.max(0, t * 1.5));
        if (radius <= 0.2) continue;
        ctx.beginPath();
        ctx.arc(x + step / 2, y + step / 2, radius, 0, Math.PI * 2);
        ctx.fillStyle = "#fc5000";
        ctx.fill();
      }
    }
  }, []);

  return (
    <canvas
      ref={ref}
      width={640}
      height={480}
      className="block h-full min-h-[340px] w-full rounded-card"
      role="img"
      aria-label="Halftone dot field, violet shading into ember orange"
    />
  );
}
