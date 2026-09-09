# HIDS-IoMT results dashboard

Next.js 16 (App Router, React 19, Tailwind v4, TypeScript), statically exported. It reads no
server and fetches nothing: every figure is compiled in from `lib/data.ts`.

```bash
npm install
npm run dev        # http://localhost:3000
npm run build      # static export to out/
npm run typecheck
```

## Where the numbers come from

`lib/data.ts` is the single source for every value on the page, and each group names the report it
was transcribed from (`../reports/*.md`, `../reports/ns3_runs/`). Nothing is estimated and nothing
is carried over from the base paper. **Attack is the positive class** throughout, per `TRD.md 2.3`.

If a report changes, change `lib/data.ts`. Do not edit figures into the components.

## Structure

| Path | What it holds |
|---|---|
| `lib/data.ts` | Every figure, typed, with its source report named |
| `app/page.tsx` | A server component composing the sections. No client code |
| `app/globals.css` | Caldera tokens from `../DESIGN.md` as Tailwind v4 `@theme` |
| `components/primitives.tsx` | Card, Tag, Section, KeyValue: the surfaces DESIGN.md defines |
| `components/CrossProduct.tsx` | Client. The 48-run NS-3 grid, one rule at a time |
| `components/Halftone.tsx` | Client. Canvas halftone, DESIGN.md's signature motif |
| `components/Nav.tsx` | Client. Section rail, IntersectionObserver rather than a scroll listener |

Three client leaves, everything else server-rendered.

## Design system

`../DESIGN.md` ("Caldera") is authoritative and this app does not deviate from it: Pumice canvas,
Limestone cards, Ember as the only aggressive accent, Plasma Violet confined to the halftone and a
single card, Sulfur only on tags, no shadows, 40px cards / 800px pills / 16px small.

DESIGN.md declares `Theme: light` and defines no dark palette, so the page commits to one visual
world rather than inventing a second one. Result state is encoded by **form** (solid fill, dotted
outline, obsidian block) as well as colour, so nothing depends on hue alone.

## `standalone/index.html`

The original single-file build, kept because an Artifact must be one self-contained HTML file and a
Next export is not. **It carries its own copy of the numbers and will drift from `lib/data.ts`.**
Treat this app as the source of truth; regenerate or retire the standalone rather than editing both.
