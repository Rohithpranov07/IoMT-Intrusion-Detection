"""Parse RESULT lines from runs/*.txt into pandas DataFrames."""
import glob
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUMERIC_SKIP = {"label", "rule", "coupling", "mitigation", "topology", "deviceMix", "weightScheme",
                "latencySource", "stage2Mode", "arch", "scenario", "factor", "branch"}


def parse_line(line):
    row = {}
    body = line.strip()[len("RESULT,"):]
    for kv in body.split(","):
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        if k not in NUMERIC_SKIP:
            try:
                v = float(v)
            except ValueError:
                pass
        row[k] = v
    return row


def load(exp):
    path = os.path.join(ROOT, "runs", f"{exp}.txt")
    rows = [parse_line(l) for l in open(path) if l.startswith("RESULT,")]
    df = pd.DataFrame(rows)
    df["experiment"] = exp
    return df


ARCH_ORDER = ["A", "B", "C", "B'hys", "B'dual", "B'cap"]
ARCH_NAME = {"A": "A flat (base paper)", "B": "B EWMA", "C": "C criticality",
             "B'hys": "B' hysteresis", "B'dual": "B' dual threshold", "B'cap": "B' capacity-coupled"}
