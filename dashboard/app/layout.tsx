import type { Metadata } from "next";
import { Anton, DM_Sans } from "next/font/google";
import "./globals.css";

/*
 * DESIGN.md specifies PP Neue Corp Compact, a licensed face this repository does not ship, and
 * names Bebas Neue / Anton / Druk Wide Bold as its substitutes. Anton is the closest of the three
 * for weight and width, and self-hosting through next/font avoids a render-blocking font request.
 */
const display = Anton({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
});

/* DESIGN.md: Medium (500) only. Regular reads as anemic against the display face; Bold competes. */
const body = DM_Sans({
  weight: ["500"],
  subsets: ["latin"],
  variable: "--font-body",
  display: "swap",
});

export const metadata: Metadata = {
  title: "HIDS-IoMT Findings",
  description:
    "Results for a critique-driven reimplementation of the HIDS-IoMT intrusion detection paper, including the negative findings.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable}`}>
      <body className="font-[family-name:var(--font-body)] bg-pumice text-obsidian">
        {children}
      </body>
    </html>
  );
}
