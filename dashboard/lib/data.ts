/**
 * Every figure rendered by this dashboard, in one typed place.
 *
 * Each value below is transcribed from a generated report in `../reports/`. Nothing here is
 * estimated, rounded in this project's favour, or carried over from the base paper. The source
 * report is named on every group so a reader can check any number against its origin.
 *
 * POSITIVE CLASS = Attack (label 1). This is TRD.md 2.3's convention and the opposite of the
 * base paper's, which uses Normal as positive (PRD.md 2.1.3).
 */

export type Provenance = "measured" | "simulated" | "blocked";

export interface MetricRow {
  model: string;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  extra?: string;
  best?: boolean;
}

export interface Stat {
  label: string;
  value: string;
  note: string;
  tone?: "ember" | "plasma" | "quiet";
}

/** Headline counters. Source: README 1, reports/ns3_simulation_results.md. */
export const STATS: Stat[] = [
  {
    label: "Test suite",
    value: "393",
    note: "373 pass, 20 skip. Skips are gated on hardware, not failures.",
    tone: "ember",
  },
  {
    label: "Duplicate rows, IoTID20",
    value: "58.2%",
    note: "The larger leakage vector. Neither the base paper nor the prior work mentions it.",
    tone: "ember",
  },
  {
    label: "Duplicate rows, Edge-IIoTset",
    value: "98.95%",
    note: "157,800 rows hold 1,661 distinct feature vectors.",
    tone: "ember",
  },
  {
    label: "NS-3 raw runs",
    value: "60",
    note: "Saved before any aggregation, so every table is re-derivable.",
    tone: "plasma",
  },
];

export interface Objection {
  verdict: string;
  title: string;
  body: string;
}

/** Source: PRD.md 2.1, README 4. */
export const OBJECTIONS: Objection[] = [
  {
    verdict: "Confirmed, then fixed",
    title: "Pre-split SMOTE inflated the headline",
    body:
      "The cleaned majority class is 585,342 rows, and 2 x 585,342 reproduces the paper's stated post-SMOTE total. Its folds partition synthetic data. Reproducing that order put 46.61% of the test fold into rows that never existed in the capture.",
  },
  {
    verdict: "Fixed",
    title: "The LSTM was never specified",
    body:
      "Neither layer nor unit count appears in the paper, and its input is a single flow record. This project states both: 2 layers, 64 units per direction, over genuine 10-step sequences. A test fails the build if the code and the decision record disagree.",
  },
  {
    verdict: "Confirmed by re-derivation",
    title: "Precision and recall labels are swapped",
    body:
      "The paper takes Normal as positive and labels both Equations 7 and 8 'Recall', one of which is the precision formula. Recomputing from its own Fig. 11c matrix reproduces the reported accuracy exactly, but shows the reported precision is the true recall.",
  },
  {
    verdict: "Partly fixed",
    title: "Black box, flat 500 ms for every device",
    body:
      "Replaced with per-alert certified SHAP and a criticality-weighted threshold. The explanation layer works. The false-positive target it was meant to hit does not, and the adaptive section says so.",
  },
];

/** Record level, IoTID20. Source: reports/t1_2_baseline_comparison.csv. */
export const RECORD_LEVEL: MetricRow[] = [
  { model: "Leaky, the base paper's order", accuracy: 0.9986, precision: 0.9975, recall: 0.9996, f1: 0.9986, extra: "46.61%" },
  { model: "Honest order, duplicates kept", accuracy: 0.9988, precision: 0.9991, recall: 0.9996, f1: 0.9994, extra: "0.00%" },
  { model: "Honest order, deduplicated", accuracy: 0.9964, precision: 0.9971, recall: 0.999, f1: 0.998, extra: "0.00%", best: true },
];

export const RECORD_LEVEL_NOTE =
  "Fixing the SMOTE ordering alone did not lower the number. The middle row is why the third exists: 58.2% of cleaned rows are exact duplicates, so a random split puts identical records on both sides no matter when SMOTE runs.";

/** Window level, IoTID20. Source: reports/t4_4_ablation.csv. */
export const IOTID20_WINDOW: MetricRow[] = [
  { model: "bilstm + transformer", accuracy: 0.9487, f1: 0.971, best: true },
  { model: "transformer alone", accuracy: 0.9487, f1: 0.9707 },
  { model: "Full three-branch ensemble", accuracy: 0.9399, f1: 0.9661 },
  { model: "bilstm alone", accuracy: 0.9271, f1: 0.9595 },
  { model: "cnn alone", accuracy: 0.923, f1: 0.9565 },
];

export const IOTID20_NOTE =
  "The best single branch beats the ensemble, and removing the CNN improves it by 0.0049.";

/** Window level, Edge-IIoTset. Source: reports/t3_6_ensemble.md. */
export const EDGE_WINDOW: MetricRow[] = [
  { model: "cnn alone", accuracy: 0.9798, f1: 0.9887, best: true },
  { model: "Full three-branch ensemble", accuracy: 0.9214, f1: 0.9537 },
  { model: "bilstm alone", accuracy: 0.9201, f1: 0.9529 },
  { model: "transformer alone", accuracy: 0.8945, f1: 0.9372 },
];

export const EDGE_NOTE =
  "The same verdict on a second dataset, by a wider margin, with a different winning branch. No branch is reliably best and the fusion does not exploit that.";

export interface RuleRow {
  rule: string;
  alerts: number;
  truePositives: number;
  falsePositives: number;
  attackersBlocked: string;
  wearablesBlocked: string;
  best?: boolean;
}

/** Source: reports/ns3_simulation_results.md 3. All SIMULATED. */
export const NS3_RULES: RuleRow[] = [
  { rule: "fixed, the base paper's flat 500 ms", alerts: 327, truePositives: 267, falsePositives: 60, attackersBlocked: "60/60", wearablesBlocked: "20/20" },
  { rule: "ewma, congestion-adaptive", alerts: 303, truePositives: 267, falsePositives: 36, attackersBlocked: "60/60", wearablesBlocked: "8/20" },
  { rule: "criticality, this project", alerts: 296, truePositives: 267, falsePositives: 29, attackersBlocked: "60/60", wearablesBlocked: "9/20", best: true },
];

export const NS3_RULES_NOTE =
  "The flat rule blocks every legitimate wearable in the simulation. A rule that quarantines the whole patient-monitoring estate has not prevented an incident, it has caused one.";

export type RuleKey = "fixed" | "ewma" | "criticality";

export interface CrossProduct {
  /** Percent of attacker-bots blocked, indexed [latency][attackScale]. */
  blocked: number[][];
  /** False-positive alerts raised against legitimate wearables, same indexing. */
  falseAlerts: number[][];
}

export const LATENCIES = ["57.9 us", "500 us", "2000 us", "14542.7 us"] as const;
export const ATTACK_SCALES = ["20 bots", "40 bots", "60 bots", "80 bots"] as const;

/**
 * The full 48-run cross product. Source: the 48 `rule-*_lat-*_atk-*.txt` runs in
 * `../reports/ns3_runs/`, parsed by `scripts/ns3_experiments/aggregate_results.py`.
 *
 * The `ewma` block is the point of running the whole cube rather than two slices through it:
 * it holds at 20 and 40 bots and collapses to zero at 60 and 80, at the SAME 2000 us latency.
 */
export const CROSS_PRODUCT: Record<RuleKey, CrossProduct> = {
  fixed: {
    blocked: [
      [100, 100, 100, 100],
      [100, 100, 100, 100],
      [100, 100, 100, 100],
      [100, 100, 100, 100],
    ],
    falseAlerts: [
      [60, 60, 60, 60],
      [60, 60, 60, 60],
      [60, 60, 60, 60],
      [64, 61, 67, 61],
    ],
  },
  ewma: {
    blocked: [
      [100, 100, 100, 100],
      [100, 100, 100, 100],
      [100, 100, 0, 0],
      [0, 0, 0, 0],
    ],
    falseAlerts: [
      [34, 31, 36, 43],
      [34, 31, 36, 43],
      [34, 31, 8, 7],
      [8, 6, 8, 7],
    ],
  },
  criticality: {
    blocked: [
      [100, 100, 100, 100],
      [100, 100, 100, 100],
      [100, 100, 100, 100],
      [100, 100, 100, 100],
    ],
    falseAlerts: [
      [30, 30, 29, 30],
      [30, 30, 29, 30],
      [30, 30, 29, 30],
      [34, 30, 33, 33],
    ],
  },
};

/** The tipping-point probe. Source: reports/ns3_runs/tn2_race_ewma_*.txt. */
export const RACE = [
  { label: "blockAfter 1", value: "60/60 blocked, 0 dropped" },
  { label: "blockAfter 2", value: "60/60 blocked, 0 dropped" },
  { label: "blockAfter 3", value: "0/60 blocked, 280,128 dropped" },
  { label: "blockAfter 4", value: "0/60 blocked, 280,128 dropped" },
];

/** Source: reports/t3_3_xai_evaluation.md, reports/t3_2_lime_vs_shap.md. */
export const XAI_STATS: Stat[] = [
  { label: "Explanations that certify", value: "82%", note: "The other 18% are labelled as unverified, not hidden.", tone: "ember" },
  { label: "SHAP stability, top-5 Jaccard", value: "0.87", note: "Under 1% input noise, while the model's own prediction flip rate is 0%.", tone: "quiet" },
  { label: "LIME stability, same test", value: "0.27", note: "Retained as a cross-check. Never shown to a human.", tone: "quiet" },
];

export interface ThresholdRow {
  mix: string;
  falsePositives: number;
  change: string;
  falseNegatives: number;
  precision: number;
  recall: number;
}

/** Source: reports/t3_4_adaptive_threshold.md. */
export const THRESHOLD_MIXES: ThresholdRow[] = [
  { mix: "Flat baseline, every device alike", falsePositives: 222, change: "0.0%", falseNegatives: 72, precision: 0.9497, recall: 0.9831 },
  { mix: "Critical-heavy, an ICU segment", falsePositives: 274, change: "+23.4%", falseNegatives: 37, precision: 0.9391, recall: 0.9913 },
  { mix: "Balanced, a mixed ward", falsePositives: 224, change: "+0.9%", falseNegatives: 74, precision: 0.9493, recall: 0.9826 },
  { mix: "Periphery-heavy, a general network", falsePositives: 198, change: "-10.8%", falseNegatives: 97, precision: 0.9546, recall: 0.9773 },
];

export const THRESHOLD_NOTE =
  "The target set in the project requirements was a greater than 30% false-positive reduction. The best measured figure is -10.8%, and in the ICU mix false positives rise by 23.4% while missed attacks fall by 48.6%. That trade is the intended behaviour, but the target is not met.";

/** Source: reports/t3_5_incremental_learning.md. */
export const INCREMENTAL = [
  { label: "Recall on the new attack type", value: "0.9746 to 0.9880" },
  { label: "Recall on prior types", value: "0.9821 to 0.9863" },
  { label: "Forgetting", value: "+0.0042" },
  { label: "Trainable parameters", value: "8,646" },
];

/** Source: reports/t4_1_export.md. */
export const DEPLOYMENT = [
  { label: "TFLite graphs, float32", value: "1.4 MB" },
  { label: "TFLite graphs, float16", value: "798 KB" },
  { label: "Prediction agreement with Keras", value: "100%" },
  { label: "Standalone Pi bundle", value: "about 2 MB" },
];

export interface Finding {
  title: string;
  body: string;
}

/** Source: README 8. */
export const FINDINGS: Finding[] = [
  {
    title: "Calibration was the obvious fix, and it does nothing",
    body:
      "Two reports had blamed miscalibration and called temperature scaling the highest-value remaining work. That was an inference, and measurement refuted it. The branches are already well calibrated at 0.012 to 0.023 expected calibration error, and the BiLSTM is mildly under-confident. Fitting the fusion priors also fails, overfitting the validation fold. Both corrections were applied at source with the original reasoning left visible.",
  },
  {
    title: "A blocker this project reported was its own reasoning error",
    body:
      "An earlier report cut half of the Edge-IIoTset task on the grounds that its records could not be ordered in time. The timestamps lose their date to an unquoted comma, but the clock survives on 90.04% of rows and confirms the file's row order rather than contradicting it, with 13 backward steps in 142,088 rows. An ordering existed the whole time. The error was inferring a property of the data from a corrupted column instead of parsing it.",
  },
  {
    title: "A fold-balancing proxy that is safe on one dataset and wrong on the next",
    body:
      "The session splitter counts each session's entries and documents them as windows; one notebook passes records. On IoTID20 that proxy holds. On Edge-IIoTset, with attack sessions of 7,050 records against a normal median of 2, it produced folds whose test set held no normal windows at all, and every model scored precision 1.0000 against zero negatives. That reads as a triumph and is an empty set.",
  },
  {
    title: "An unpinned install silently changed the environment",
    body:
      "A single dependency install upgraded numpy and scikit-learn past the versions the requirements file already named. It surfaced only when an unrelated package stopped importing, several tasks later. Every package is now pinned exactly and a test asserts the live environment matches.",
  },
];

export interface OpenItem {
  status: string;
  title: string;
  body: string;
}

/** Source: README 7. */
export const OPEN_ITEMS: OpenItem[] = [
  {
    status: "Needs hardware",
    title: "Pi latency and throughput",
    body: "The export, the standalone runner and the bundle are all verified. Only the measurement is missing.",
  },
  {
    status: "Needs a person",
    title: "The reader test",
    body:
      "A teammate unfamiliar with the model must read one alert and say why the flow was flagged. It cannot be self-assessed. One alert in the handout is deliberately unverified, and whether they notice is the real result.",
  },
  {
    status: "Not attempted",
    title: "Cross-dataset validation",
    body:
      "CICIDS2017 was scoped as a should and was not attempted. Every result here is a single seed and a single split, which is an omission this project criticises in the prior work and then inherits.",
  },
];

export const SECTIONS = [
  { id: "objections", label: "Objections" },
  { id: "detection", label: "Detection" },
  { id: "simulation", label: "Simulation" },
  { id: "explain", label: "Explainability" },
  { id: "adaptive", label: "Adaptive" },
  { id: "deployment", label: "Deployment" },
  { id: "negative", label: "Negative results" },
  { id: "open", label: "Open" },
] as const;
