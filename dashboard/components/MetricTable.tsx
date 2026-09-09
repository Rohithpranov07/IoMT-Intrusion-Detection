import type { MetricRow } from "@/lib/data";
import { Caption } from "./primitives";

/**
 * A metrics table that cannot be copied out of this page without its positive-class convention.
 * Wide content scrolls inside its own container so the page body never scrolls sideways, and the
 * caption sits OUTSIDE that container: inside it, the scroll box clips its first few pixels.
 */
export function MetricTable({
  rows,
  columns,
  note,
  extraHeader,
}: {
  rows: MetricRow[];
  columns: Array<"accuracy" | "precision" | "recall" | "f1">;
  note?: string;
  extraHeader?: string;
}) {
  const labels: Record<string, string> = {
    accuracy: "Accuracy",
    precision: "Precision",
    recall: "Recall",
    f1: "F1",
  };

  return (
    <>
      <div className="overflow-x-auto rounded-medium">
        <table className="w-full min-w-[520px] border-collapse text-sm">
          <caption className="sr-only">
            Attack is the positive class. Every row is hand-verified against its own confusion
            matrix.
          </caption>
          <thead>
            <tr className="border-b-[1.5px] border-obsidian">
              <th scope="col" className="px-3.5 py-3 text-left text-xs uppercase tracking-[0.06em]">
                Model
              </th>
              {columns.map((c) => (
                <th
                  key={c}
                  scope="col"
                  className="px-3.5 py-3 text-right text-xs uppercase tracking-[0.06em] whitespace-nowrap"
                >
                  {labels[c]}
                </th>
              ))}
              {extraHeader ? (
                <th
                  scope="col"
                  className="px-3.5 py-3 text-right text-xs uppercase tracking-[0.06em] whitespace-nowrap"
                >
                  {extraHeader}
                </th>
              ) : null}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.model}
                className={
                  row.best
                    ? "bg-ember text-chalk"
                    : "border-t border-dotted border-obsidian/30"
                }
              >
                <th
                  scope="row"
                  className={`px-3.5 py-3 text-left font-medium min-w-[200px] ${
                    row.best ? "rounded-l-small" : ""
                  }`}
                >
                  {row.model}
                </th>
                {columns.map((c) => (
                  <td key={c} className="num px-3.5 py-3 text-right whitespace-nowrap">
                    {row[c]?.toFixed(4)}
                  </td>
                ))}
                {extraHeader ? (
                  <td
                    className={`num px-3.5 py-3 text-right whitespace-nowrap ${
                      row.best ? "rounded-r-small" : ""
                    }`}
                  >
                    {row.extra}
                  </td>
                ) : null}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {note ? <Caption>{note}</Caption> : null}
    </>
  );
}
