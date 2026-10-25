"""Review 3 experiment definitions and resumable runner.

Every configuration is a dict of simulator flags. Each finished run appends its single
'RESULT,' line (key=value pairs, see sim/hids-iomt-r3.cc) to runs/<experiment>.csv-like
.txt files, so the raw output of every run is kept before any aggregation, and a rerun skips
labels that already exist (the runner is resumable after an interruption).

Usage:  python3 scripts/experiments.py E1 E2a E3 ...   (runs the named experiments in order)
"""
import itertools
import multiprocessing as mp
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "sim", "hids-r3")
RUNS = os.path.join(ROOT, "runs")
os.makedirs(RUNS, exist_ok=True)

SEEDS20 = list(range(1, 21))
SEEDS10 = list(range(1, 11))

# Architectures. B-prime variants are EWMA plus one latch-off mitigation (Phase 3).
ARCH = {
    "A": {"rule": "fixed"},
    "B": {"rule": "ewma"},
    "C": {"rule": "criticality"},
    "B'hys": {"rule": "ewma", "mitigation": "hysteresis"},
    "B'dual": {"rule": "ewma", "mitigation": "dual"},
    "B'cap": {"rule": "ewma", "mitigation": "capacity"},
}
BASE = {"nWearables": 20, "nAttackers": 60, "nFog": 4, "topology": "star", "deviceMix": "balanced",
        "idsLatencyUs": 500, "blockAfter": 3, "simTime": 30, "flowmon": 0,
        "latencySource": "SWEPT-no-hardware-measurement"}

# Phase 1.2 one-factor-at-a-time deviations from BASE (BASE itself is E1).
SCENARIOS = (
    [("wear", {"nWearables": v}) for v in (10, 40, 80)]
    + [("bots", {"nAttackers": v}) for v in (20, 40, 80, 120)]
    + [("fog", {"nFog": v}) for v in (1, 2, 8)]
    + [("topo", {"topology": v}) for v in ("clustered", "grid", "random")]
    + [("mix", {"deviceMix": v}) for v in ("icu", "periphery")]
)

# Measured stage-2 operating points from t4_4_ablation.csv (IoTID20 windows, Attack positive):
# TPR = TP/(TP+FN), FPR = FP/(FP+TN).
STAGE2 = {
    "bilstm": (4229 / (4229 + 36), 321 / (321 + 308)),
    "transformer": (4152 / (4152 + 113), 138 / (138 + 491)),
    "ensemble": (4193 / (4193 + 72), 222 / (222 + 407)),
}


def label_of(exp, cfg):
    keys = sorted(k for k in cfg if k not in ("latencySource", "flowmon", "label"))
    return exp + "|" + "|".join(f"{k}={cfg[k]}" for k in keys)


def mk(exp, arch, seed, **over):
    cfg = dict(BASE)
    cfg.update(ARCH[arch])
    cfg.update(over)
    cfg["rngRun"] = seed
    cfg["arch"] = arch
    return cfg


def build(exp):
    out = []
    if exp == "E1":  # Phase 1.1 baseline, both inspection costs
        for lat, a, s in itertools.product((500, 2000), ARCH, SEEDS20):
            out.append(mk(exp, a, s, idsLatencyUs=lat))
    elif exp in ("E2a", "E2b"):  # Phase 1.2 matrix at 500 us (a) and 2000 us (b)
        lat = 500 if exp == "E2a" else 2000
        for (name, dev), a in itertools.product(SCENARIOS, ARCH):
            # A, B, C: 20 seeds at 500 us. B' variants (identical to B whenever the queue is not
            # saturated, see E1) and the 2000 us matrix: 10 seeds. Baseline E1 has 20 for all six.
            seeds = SEEDS10 if exp == "E2a" else (1, 2, 3, 4, 5)  # compute budget; baseline E1 keeps 20
            for s in seeds:
                    out.append(mk(exp, a, s, idsLatencyUs=lat, scenario=f"{name}={list(dev.values())[0]}", **dev))
    elif exp == "E3":  # Phase 1.3 sensitivity, 20 seeds
        for v, a, s in itertools.product((100, 250, 1000), ("A", "B", "C"), SEEDS10):
            out.append(mk(exp, a, s, baseMs=v, factor="tau0"))
        for v, lat, s in itertools.product((0.1, 0.5, 0.7), (500, 2000), SEEDS10):
            out.append(mk(exp, "B", s, ewmaAlpha=v, idsLatencyUs=lat, factor="alpha"))
        for v, s in itertools.product(("uniform", "linear", "exponential"), SEEDS10):
            out.append(mk(exp, "C", s, weightScheme=v, factor="weights"))
        for v, a, s in itertools.product((57.9, 1000, 14542.7), ("A", "B", "C"), SEEDS10):
            out.append(mk(exp, a, s, idsLatencyUs=v, factor="cost"))
        # patience is swept jointly with cost, attack scale and alpha in E4 (phase diagram)
    elif exp == "E4":  # Phase 3.1 latch-off phase diagram (EWMA rule as published in Review 2)
        for c, n, p, al, s in itertools.product((500, 1000, 2000, 3000, 5000), (20, 40, 60, 80, 120),
                                                (1, 2, 3, 4, 5), (0.1, 0.3, 0.7), (1,)):
            out.append(mk(exp, "B", s, idsLatencyUs=c, nAttackers=n, blockAfter=p, ewmaAlpha=al, flowmon=0, simTime=20))
    elif exp == "E5":  # Phase 4.1 trade-off curve: half the bots are slow mimics (300-700 ms)
        taus = (50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 1000, 1200, 1500)
        for t, a, s in itertools.product(taus, ("A", "B", "C"), (1, 2, 3, 4, 5)):
            out.append(mk(exp, a, s, baseMs=t, slowFraction=0.5, slowMinMs=300, slowMaxMs=700, flowmon=0))
    elif exp == "E6":  # Phase 4.2 two-stage: criticality AND a measured classifier branch
        taus = (100, 300, 500, 1000, 1500)
        for br, mode, t, s in itertools.product(("bilstm", "transformer"), ("independent", "correlated"), taus, (1, 2, 3, 4, 5)):
            tpr, fpr = STAGE2[br]
            out.append(mk(exp, "C", s, baseMs=t, slowFraction=0.5, slowMinMs=300, slowMaxMs=700, flowmon=0,
                          twoStage=1, stage2Tpr=round(tpr, 6), stage2Fpr=round(fpr, 6), stage2Mode=mode,
                          branch=br))
    elif exp == "E7":  # Phase 4.2 headline: two-stage on the Review 2 baseline scenario, 20 seeds
        for br, mode, s in itertools.product(STAGE2, ("independent", "correlated"), SEEDS20):
            tpr, fpr = STAGE2[br]
            out.append(mk(exp, "C", s, twoStage=1, stage2Tpr=round(tpr, 6), stage2Fpr=round(fpr, 6),
                          stage2Mode=mode, branch=br))
    elif exp == "E8":  # Phase 5.2: Table 5 re-run with MEASURED per-flow costs (x86 vCPU, LiteRT)
        import json as _json
        lat = _json.load(open(os.path.join(ROOT, "results", "phase5", "latency_x86.json")))["per_flow_cost_us_for_ns3"]
        costs = {"x86-idle-p50": round(lat["idle_p50_us"], 1), "x86-load-p95": round(lat["full_load_p95_us"], 1)}
        for (src, c), a, n, s in itertools.product(costs.items(), ("A", "B", "C", "B'hys"), (20, 40, 60, 80), SEEDS10):
            if a == "B'hys" and src != "x86-load-p95":
                continue
            out.append(mk(exp, a, s, idsLatencyUs=c, nAttackers=n, flowmon=0,
                          latencySource=f"MEASURED-{src}-IntelXeon2.1GHz-2vCPU-LiteRT-float32-NOT-A-PI",
                          factor=src))
    elif exp == "E9":  # Phase 5: Table 5 re-run with costs MEASURED ON THE RASPBERRY PI 4 (trained models)
        import json as _json
        pi = _json.load(open(os.path.join(ROOT, "results", "phase5", "latency_pi.json")))
        assert pi["host"]["is_raspberry_pi"], "latency_pi.json was not measured on a Raspberry Pi"
        lat = pi["per_flow_cost_us_for_ns3"]
        costs = {"pi4-idle-p50": round(lat["idle_p50_us"], 1), "pi4-load-p95": round(lat["full_load_p95_us"], 1)}
        board = pi["host"]["board"].replace(" ", "")
        for (src, c), a, n, s in itertools.product(costs.items(), ("A", "B", "C", "B'hys"), (20, 40, 60, 80), SEEDS10):
            out.append(mk(exp, a, s, idsLatencyUs=c, nAttackers=n, flowmon=0,
                          latencySource=f"MEASURED-{src}-{board}-LiteRT-float32-trained-ensemble",
                          factor=src))
    else:
        raise SystemExit(f"unknown experiment {exp}")
    for c in out:
        c["label"] = label_of(exp, c)
    return out


SIM_FLAGS = None


def to_args(cfg):
    skip = {"arch", "scenario", "factor", "branch"}
    return [f"--{k}={v}" for k, v in cfg.items() if k not in skip]


def run_one(cfg):
    try:
        p = subprocess.run([BIN] + to_args(cfg), capture_output=True, text=True, timeout=900)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT,")), None)
        if line is None:
            return cfg, None, (p.stderr or p.stdout)[-500:]
        extra = ",".join(f"{k}={cfg[k]}" for k in ("arch", "scenario", "factor", "branch") if k in cfg)
        return cfg, line + ("," + extra if extra else ""), None
    except Exception as e:  # noqa: BLE001
        return cfg, None, repr(e)


def done_labels(path):
    if not os.path.exists(path):
        return set()
    out = set()
    for l in open(path):
        for kv in l.strip().split(","):
            if kv.startswith("label="):
                out.add(kv[6:])
    return out


def main(exps, procs=2):
    for exp in exps:
        cfgs = build(exp)
        path = os.path.join(RUNS, f"{exp}.txt")
        have = done_labels(path)
        todo = [c for c in cfgs if c["label"] not in have]
        print(f"[{exp}] {len(cfgs)} configs, {len(todo)} to run", flush=True)
        with mp.Pool(procs) as pool, open(path, "a") as f, open(os.path.join(RUNS, "errors.log"), "a") as ef:
            for i, (cfg, line, err) in enumerate(pool.imap_unordered(run_one, todo), 1):
                if line:
                    f.write(line + "\n")
                    f.flush()
                else:
                    ef.write(f"{cfg['label']} :: {err}\n")
                    ef.flush()
                if i % 50 == 0:
                    print(f"[{exp}] {i}/{len(todo)}", flush=True)
        print(f"[{exp}] finished", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
