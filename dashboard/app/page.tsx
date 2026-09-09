import { CrossProduct } from "@/components/CrossProduct";
import { Halftone } from "@/components/Halftone";
import { MetricTable } from "@/components/MetricTable";
import { Nav } from "@/components/Nav";
import { StatCard } from "@/components/StatCard";
import {
  Caption,
  Card,
  KeyValue,
  Section,
  SectionHead,
  Tag,
} from "@/components/primitives";
import {
  DEPLOYMENT,
  EDGE_NOTE,
  EDGE_WINDOW,
  FINDINGS,
  INCREMENTAL,
  IOTID20_NOTE,
  IOTID20_WINDOW,
  NS3_RULES,
  NS3_RULES_NOTE,
  OBJECTIONS,
  OPEN_ITEMS,
  RACE,
  RECORD_LEVEL,
  RECORD_LEVEL_NOTE,
  STATS,
  THRESHOLD_MIXES,
  THRESHOLD_NOTE,
  XAI_STATS,
} from "@/lib/data";

/**
 * Server component. Every figure is read from `lib/data.ts` at build time; nothing on this page
 * is fetched, and the only client code is the halftone canvas, the nav scrollspy and the cross
 * product selector.
 */
export default function Page() {
  return (
    <>
      <Nav />

      <header className="mx-auto grid max-w-[1280px] items-end gap-8 px-4 pt-8 sm:gap-10 sm:px-6 sm:pt-12 md:grid-cols-[1.15fr_0.85fr]">
        <div className="flex min-w-0 flex-col gap-6">
          <div className="flex flex-wrap gap-2">
            <Tag>Attack = positive class</Tag>
            <Tag>373 pass / 20 skip</Tag>
          </div>
          <h1 className="display text-[clamp(48px,7.6vw,96px)] leading-[0.95]">
            Four defects, and what survived them.
          </h1>
          <p className="max-w-[62ch] text-lg leading-relaxed">
            A critique-driven reimplementation of HIDS-IoMT (Berguiga et al., IEEE Access 13,
            2025). It does not try to beat the paper&rsquo;s 99.92%. It reproduces the leakage that
            produced it, then reports every result that came out negative, of which there are
            several.
          </p>
        </div>
        <Halftone />
      </header>

      <main className="mx-auto max-w-[1280px] px-4 sm:px-6">
        <Section>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {STATS.map((stat) => (
              <StatCard key={stat.label} stat={stat} />
            ))}
          </div>
        </Section>

        <Section id="objections">
          <SectionHead
            title="The four objections"
            lede="Each was stated before the work began. Two were confirmed, one was fixed, one only partly."
          />
          <div className="grid gap-4 md:grid-cols-2">
            {OBJECTIONS.map((objection) => (
              <Card key={objection.title}>
                <Tag>{objection.verdict}</Tag>
                <h3 className="display mt-4 text-[26px] leading-tight">{objection.title}</h3>
                <p className="mt-3">{objection.body}</p>
              </Card>
            ))}
          </div>
        </Section>

        <Section id="detection">
          <SectionHead
            title="Detection results"
            lede="Attack is the positive class in every table. Each row is hand-verified against its own confusion matrix."
          />

          <Card>
            <h3 className="display text-[26px] leading-tight">Record level, IoTID20</h3>
            <div className="mt-6">
              <MetricTable
                rows={RECORD_LEVEL}
                columns={["accuracy", "precision", "recall", "f1"]}
                extraHeader="Test fold synthetic"
                note={RECORD_LEVEL_NOTE}
              />
            </div>
          </Card>

          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <Card>
              <h3 className="display text-[26px] leading-tight">Window level, IoTID20</h3>
              <div className="mt-6">
                <MetricTable
                  rows={IOTID20_WINDOW}
                  columns={["accuracy", "f1"]}
                  note={IOTID20_NOTE}
                />
              </div>
            </Card>
            <Card>
              <h3 className="display text-[26px] leading-tight">Window level, Edge-IIoTset</h3>
              <div className="mt-6">
                <MetricTable rows={EDGE_WINDOW} columns={["accuracy", "f1"]} note={EDGE_NOTE} />
              </div>
            </Card>
          </div>

          <Card tone="plasma" className="mt-4">
            <h3 className="display text-[26px] leading-tight">
              The ensemble does not earn its complexity
            </h3>
            <p className="mt-4 max-w-[70ch]">
              The three branches agree on 90.9% of IoTID20 windows, so fusion can act on 9.1%, and
              on those contested windows it is worse than its best branch. An oracle resolving every
              disagreement perfectly would gain only 0.026 accuracy. This is the project&rsquo;s
              central negative result and it is reported first, not buried.
            </p>
          </Card>
        </Section>

        <Section id="simulation">
          <SectionHead
            title="Network simulation"
            lede="NS-3, 20 legitimate wearables against 60 attacker-bots across 4 fog nodes. Every number here is simulation output. No Raspberry Pi measurement exists anywhere in this project, and none is implied."
          />

          <Card>
            <h3 className="display text-[26px] leading-tight">Three threshold rules, one scenario</h3>
            <div className="mt-6 overflow-x-auto rounded-medium">
              <table className="w-full min-w-[640px] border-collapse text-sm">
                <thead>
                  <tr className="border-b-[1.5px] border-obsidian">
                    {["Rule", "Alerts", "True positives", "False positives", "Attackers blocked", "Wearables blocked"].map(
                      (heading, index) => (
                        <th
                          key={heading}
                          scope="col"
                          className={`px-3.5 py-3 text-xs uppercase tracking-[0.06em] whitespace-nowrap ${
                            index === 0 ? "text-left" : "text-right"
                          }`}
                        >
                          {heading}
                        </th>
                      )
                    )}
                  </tr>
                </thead>
                <tbody>
                  {NS3_RULES.map((row) => (
                    <tr
                      key={row.rule}
                      className={
                        row.best ? "bg-ember text-chalk" : "border-t border-dotted border-obsidian/30"
                      }
                    >
                      <th
                        scope="row"
                        className={`min-w-[220px] px-3.5 py-3 text-left font-medium ${
                          row.best ? "rounded-l-small" : ""
                        }`}
                      >
                        {row.rule}
                      </th>
                      <td className="num px-3.5 py-3 text-right">{row.alerts}</td>
                      <td className="num px-3.5 py-3 text-right">{row.truePositives}</td>
                      <td className="num px-3.5 py-3 text-right">{row.falsePositives}</td>
                      <td className="num px-3.5 py-3 text-right">{row.attackersBlocked}</td>
                      <td
                        className={`num px-3.5 py-3 text-right ${row.best ? "rounded-r-small" : ""}`}
                      >
                        {row.wearablesBlocked}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Caption>{NS3_RULES_NOTE}</Caption>
          </Card>

          <Card className="mt-4">
            <h3 className="display text-[26px] leading-tight">Where each rule goes blind</h3>
            <p className="mt-3 max-w-[70ch]">
              48 runs, every combination of rule, inspection cost and attack scale. Two slices
              through this cube would have hidden its main result: the blind spot is not at a
              latency, it is at an offered load.
            </p>
            <div className="mt-6">
              <CrossProduct />
            </div>
          </Card>

          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <Card>
              <h3 className="display text-[26px] leading-tight">
                One violation of patience decides it
              </h3>
              <p className="mt-3">
                Same rule, same latency, same 60 attacker-bots. The only variable is how many
                violations the rule waits for before quarantining a source.
              </p>
              <div className="mt-5">
                {RACE.map((row) => (
                  <KeyValue key={row.label} label={row.label} value={row.value} />
                ))}
              </div>
            </Card>
            <Card tone="ember">
              <Tag>Fifth objection</Tag>
              <h3 className="display mt-4 text-[26px] leading-tight">
                It does not degrade. It latches off.
              </h3>
              <p className="mt-3">
                Blocking removes a bot&rsquo;s load, so blocking early keeps the queue empty, which
                keeps the threshold high, which keeps blocking possible. Miss that window and the
                loop runs backwards: the queue saturates, the threshold collapses toward zero,
                nothing can violate zero, and it never recovers. A congestion-coupled rule is a
                congestion detector wearing an intrusion detector&rsquo;s label.
              </p>
            </Card>
          </div>
        </Section>

        <Section id="explain">
          <SectionHead
            title="Explainability"
            lede="A method being reliable on average is not the same as this alert being sound. Every explanation is certified before an operator sees it."
          />
          <div className="grid gap-4 md:grid-cols-3">
            {XAI_STATS.map((stat) => (
              <StatCard key={stat.label} stat={stat} />
            ))}
          </div>
          <Card className="mt-4">
            <p className="max-w-[75ch]">
              LIME&rsquo;s apparent quality was noise. Raising its sample count does not rescue it,
              and its fidelity collapses as sampling converges, from +0.349 down to +0.023. In 88%
              of its alerts the top-cited feature argues against the alert&rsquo;s own verdict. That
              is the finding that demoted it from operator-facing fallback to internal cross-check.
            </p>
          </Card>
        </Section>

        <Section id="adaptive">
          <SectionHead
            title="Adaptive threshold and incremental learning"
            lede="The criticality weighting reallocates false positives toward the devices that can afford them. It does not reduce them enough."
          />
          <Card>
            <div className="overflow-x-auto rounded-medium">
              <table className="w-full min-w-[640px] border-collapse text-sm">
                <thead>
                  <tr className="border-b-[1.5px] border-obsidian">
                    {["Device mix", "False positives", "Change", "False negatives", "Precision", "Recall"].map(
                      (heading, index) => (
                        <th
                          key={heading}
                          scope="col"
                          className={`px-3.5 py-3 text-xs uppercase tracking-[0.06em] whitespace-nowrap ${
                            index === 0 ? "text-left" : "text-right"
                          }`}
                        >
                          {heading}
                        </th>
                      )
                    )}
                  </tr>
                </thead>
                <tbody>
                  {THRESHOLD_MIXES.map((row) => (
                    <tr key={row.mix} className="border-t border-dotted border-obsidian/30">
                      <th scope="row" className="min-w-[220px] px-3.5 py-3 text-left font-medium">
                        {row.mix}
                      </th>
                      <td className="num px-3.5 py-3 text-right">{row.falsePositives}</td>
                      <td className="num px-3.5 py-3 text-right">{row.change}</td>
                      <td className="num px-3.5 py-3 text-right">{row.falseNegatives}</td>
                      <td className="num px-3.5 py-3 text-right">{row.precision.toFixed(4)}</td>
                      <td className="num px-3.5 py-3 text-right">{row.recall.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Caption>{THRESHOLD_NOTE}</Caption>
          </Card>

          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <Card>
              <h3 className="display text-[26px] leading-tight">Incremental update, replay buffer</h3>
              <div className="mt-5">
                {INCREMENTAL.map((row) => (
                  <KeyValue key={row.label} label={row.label} value={row.value} />
                ))}
              </div>
            </Card>
            <Card>
              <h3 className="display text-[26px] leading-tight">
                The rejected alternative, measured anyway
              </h3>
              <p className="mt-3">
                Fine-tuning the fusion layer alone gives three adjustable parameters, which can only
                re-mix opinions the branches already hold. Measured, it drops prior-class recall and
                pushes the false-positive rate from 0.5231 to 0.6280. It fails the gate, and the
                comparison is reported rather than asserted.
              </p>
            </Card>
          </div>
        </Section>

        <Section id="deployment">
          <SectionHead title="Deployment" />
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <h3 className="display text-[26px] leading-tight">What exists</h3>
              <div className="mt-5">
                {DEPLOYMENT.map((row) => (
                  <KeyValue key={row.label} label={row.label} value={row.value} />
                ))}
              </div>
              <p className="mt-5">
                The BiLSTM does not convert as trained. Keras exports it as a dynamic TensorList
                loop, and one standard escape produced a 16 KB file that loaded, emitted NaN, and
                agreed with the Keras model on 0% of windows. Fixed by unrolling, valid only because
                the window length is frozen at 10.
              </p>
            </Card>
            <Card tone="obsidian" className="flex flex-col gap-4">
              <Tag>Blocked on hardware</Tag>
              <h3 className="display text-[26px] leading-tight">No latency number exists</h3>
              <p>
                There is no inference latency or throughput figure anywhere in this project, because
                the Raspberry Pi 4B has not been connected. The benchmark raises an error rather
                than writing a report off-device, so the prohibition lives in code rather than in
                discipline.
              </p>
              <p>
                The objection this project makes against the prior work is a real-time claim with no
                hardware behind it. Publishing a desktop number here would be the same failure
                wearing this project&rsquo;s name.
              </p>
            </Card>
          </div>
        </Section>

        <Section id="negative">
          <SectionHead
            title="Results that came out negative"
            lede="Kept because they were expensive to learn and are the most useful thing here for anyone building on this work."
          />
          <div className="grid gap-4">
            {FINDINGS.map((finding) => (
              <Card key={finding.title}>
                <h3 className="display text-[26px] leading-tight">{finding.title}</h3>
                <p className="mt-3 max-w-[85ch]">{finding.body}</p>
              </Card>
            ))}
          </div>
        </Section>

        <Section id="open">
          <SectionHead title="Still open" lede="Three items. None is unwritten code." />
          <div className="grid gap-4 md:grid-cols-3">
            {OPEN_ITEMS.map((item) => (
              <Card key={item.title}>
                <Tag>{item.status}</Tag>
                <h3 className="display mt-4 text-[26px] leading-tight">{item.title}</h3>
                <p className="mt-3">{item.body}</p>
              </Card>
            ))}
          </div>
        </Section>
      </main>

      <footer className="mx-auto max-w-[1280px] px-4 pb-12 pt-14 sm:px-6 sm:pb-16 sm:pt-20">
        <hr className="border-0 border-t-[1.5px] border-dotted border-obsidian" />
        <p className="mt-6 max-w-[75ch]">
          The base paper&rsquo;s 99.92% accuracy, 99.91% precision, 99.99% recall and 99.95% F1 must
          not be quoted without two caveats: its precision and recall labels are very likely
          swapped, and its evaluation fold was contaminated. Every figure on this page uses Attack
          as the positive class and is hand-verified against its own confusion matrix.
        </p>
        <p className="mt-6 text-xs uppercase tracking-[0.08em]">
          Computer Networks project. Sanhit, Malika, Nehaa, Rakshan, Rohith.
        </p>
      </footer>
    </>
  );
}
