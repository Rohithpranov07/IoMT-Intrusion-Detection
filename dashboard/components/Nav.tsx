"use client";

import { useEffect, useRef, useState } from "react";
import { SECTIONS } from "@/lib/data";

/**
 * The Limestone nav pill from DESIGN.md, with the current section marked.
 *
 * Scroll position is read with IntersectionObserver rather than a scroll listener, so nothing
 * runs on every scroll frame.
 *
 * Below the breakpoint where 8 links plus the brand no longer fit on one line, `flex-wrap` would
 * break the pill into a ragged multi-row shape. Instead the link row becomes its own horizontally
 * scrollable strip (brand stays fixed, links scroll), which keeps the single-line nav rule and
 * the pill silhouette intact at every width. The active link auto-scrolls into view so a reader
 * who has scrolled the page can still see where they are without hunting sideways.
 */
export function Nav() {
  const [current, setCurrent] = useState<string>("");
  const linkRefs = useRef<Record<string, HTMLAnchorElement | null>>({});

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

  useEffect(() => {
    linkRefs.current[current]?.scrollIntoView({
      behavior: "smooth",
      inline: "center",
      block: "nearest",
    });
  }, [current]);

  return (
    <nav className="sticky top-0 z-40 bg-pumice py-3 sm:py-4">
      <div className="mx-auto max-w-[1280px] px-4 sm:px-6">
        <div className="flex items-center gap-2 rounded-pill bg-limestone py-2 pl-4 pr-2 sm:gap-2.5 sm:px-5 sm:py-2.5">
          <span className="display flex shrink-0 items-center gap-2 text-lg sm:gap-2.5 sm:text-[22px]">
            <svg width="20" height="20" viewBox="0 0 22 22" aria-hidden="true" className="shrink-0 sm:h-[22px] sm:w-[22px]">
              <path d="M11 2 20 19H2Z" fill="#fc5000" />
              <path d="M11 8.5 15.5 16h-9Z" fill="#070607" />
            </svg>
            <span>HIDS-IoMT</span>
          </span>
          <div className="no-scrollbar flex min-w-0 flex-1 items-center gap-1 overflow-x-auto sm:flex-wrap sm:justify-end sm:gap-2.5 sm:overflow-visible">
            {SECTIONS.map((section) => (
              <a
                key={section.id}
                ref={(el) => {
                  linkRefs.current[section.id] = el;
                }}
                href={`#${section.id}`}
                aria-current={current === section.id ? "true" : undefined}
                className="shrink-0 whitespace-nowrap rounded-pill px-3 py-2 text-sm no-underline transition-colors hover:bg-ember hover:text-chalk aria-[current]:bg-obsidian aria-[current]:text-chalk sm:py-1.5"
              >
                {section.label}
              </a>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
}
