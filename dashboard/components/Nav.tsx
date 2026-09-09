"use client";

import { useEffect, useState } from "react";
import { SECTIONS } from "@/lib/data";

/**
 * The Limestone nav pill from DESIGN.md, with the current section marked.
 *
 * Scroll position is read with IntersectionObserver rather than a scroll listener, so nothing
 * runs on every scroll frame.
 */
export function Nav() {
  const [current, setCurrent] = useState<string>("");

  useEffect(() => {
    if (typeof IntersectionObserver === "undefined") return;

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) setCurrent(entry.target.id);
        }
      },
      { rootMargin: "-20% 0px -70% 0px" }
    );

    const sections = document.querySelectorAll("section[id]");
    sections.forEach((section) => observer.observe(section));
    return () => observer.disconnect();
  }, []);

  return (
    <nav className="sticky top-0 z-40 bg-pumice py-4">
      <div className="mx-auto max-w-[1280px] px-6">
        <div className="flex flex-wrap items-center gap-2.5 rounded-pill bg-limestone px-5 py-2.5">
          <span className="display mr-auto flex items-center gap-2.5 text-[22px]">
            <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
              <path d="M11 2 20 19H2Z" fill="#fc5000" />
              <path d="M11 8.5 15.5 16h-9Z" fill="#070607" />
            </svg>
            HIDS-IoMT
          </span>
          {SECTIONS.map((section) => (
            <a
              key={section.id}
              href={`#${section.id}`}
              aria-current={current === section.id ? "true" : undefined}
              className="whitespace-nowrap rounded-pill px-3 py-1.5 text-sm no-underline transition-colors hover:bg-ember hover:text-chalk aria-[current]:bg-obsidian aria-[current]:text-chalk"
            >
              {section.label}
            </a>
          ))}
        </div>
      </div>
    </nav>
  );
}
