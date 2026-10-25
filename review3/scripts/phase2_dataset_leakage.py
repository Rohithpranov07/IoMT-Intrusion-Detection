"""Phase 2.2: duplicate records and train/test twin leakage across datasets.

For each dataset this measures, in a funnel of feature spaces from 'every column' to 'the space
a model actually sees':
  - exact duplicate rows (% of rows that repeat an earlier row in that space)
  - distinct feature vectors
  - label-conflicting vectors (one feature vector carrying both labels)
  - TWIN LEAKAGE: under a random stratified 80/20 split, the % of test rows whose exact feature
    vector also occurs in the training fold (mean and std over 20 split seeds). This is the
    quantity that matters for evaluation: a test row with a training twin is graded on memory.

The Edge-IIoTset funnel reproduces src/preprocessing/clean.py::clean_edge_iiotset exactly
(column lists copied verbatim), so its last step is the project's 30-feature modelling space.
IoTID20 is not reachable from this machine; its figures are quoted from the project's own
measurement in clean.py and are labelled as such in the output.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "phase2")
os.makedirs(OUT, exist_ok=True)

# --- verbatim from clean.py -----------------------------------------------------------------
EDGE_LABEL = ("Attack_label", "Attack_type")
PLACEHOLDER_TOKENS = {"0", "0.0", "0.00", "", "nan", "NaN", "None", "none"}
EDGE_ID = ("frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4")
EDGE_NONCE = ("tcp.seq", "tcp.ack", "tcp.ack_raw", "tcp.checksum", "icmp.checksum", "icmp.seq_le",
              "icmp.transmit_timestamp", "udp.stream", "dns.qry.name", "tcp.srcport", "tcp.dstport")
EDGE_CONTENT = ("tcp.payload", "tcp.options", "http.file_data", "http.request.uri.query",
                "http.request.full_uri", "http.referer", "mqtt.msg")
EDGE_ZV = ("icmp.unused", "http.tls_port", "dns.qry.type", "dns.retransmit_request_in",
           "mqtt.msg_decoded_as", "mbtcp.len", "mbtcp.trans_id", "mbtcp.unit_id")


def edge_model_space(df):
    """clean_edge_iiotset steps 1-4, returning the numeric feature matrix (no row drops)."""
    norm = df.copy()
    for c in norm.columns:
        if c not in EDGE_LABEL and norm[c].dtype == object:
            t = norm[c].astype(str).str.strip()
            norm[c] = t.where(~t.isin(PLACEHOLDER_TOKENS), "__MISSING__")
    drop = set(EDGE_LABEL) | set(EDGE_ID) | set(EDGE_NONCE) | set(EDGE_CONTENT) | set(EDGE_ZV)
    cols = [c for c in norm.columns if c not in drop]
    X = pd.DataFrame(index=norm.index)
    for c in cols:
        num = pd.to_numeric(norm[c], errors="coerce")
        X[c] = num.fillna(0.0) if num.notna().mean() > 0.99 else norm[c].astype("category").cat.codes.astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X[X.notna().all(axis=1)]


def measure(name, space, X, y, seeds=20):
    key = pd.util.hash_pandas_object(X, index=False).to_numpy()
    n = len(key)
    dup = int(pd.Series(key).duplicated().sum())
    distinct = n - dup
    lab = pd.DataFrame({"k": key, "y": y.to_numpy()})
    conflict = int((lab.groupby("k")["y"].nunique() > 1).sum())
    twin = []
    for s in range(1, seeds + 1):
        tr, te = train_test_split(np.arange(n), test_size=0.2, random_state=s, stratify=y.to_numpy())
        trk = set(key[tr])
        twin.append(100.0 * np.mean([k in trk for k in key[te]]))
    row = {"dataset": name, "feature_space": space, "rows": n, "features": X.shape[1],
           "duplicate_rows": dup, "duplicate_pct": 100.0 * dup / n, "distinct_vectors": distinct,
           "label_conflicting_vectors": conflict, "test_rows_with_train_twin_pct_mean": float(np.mean(twin)),
           "test_rows_with_train_twin_pct_std": float(np.std(twin, ddof=1))}
    print(json.dumps(row))
    return row


def main():
    rows = []

    # --- Edge-IIoTset (ML file) ------------------------------------------------------------------
    edge = pd.read_csv(os.path.join(ROOT, "data", "edge_iiotset", "ML-EdgeIIoT-dataset.csv"), low_memory=False)
    y_e = edge["Attack_label"].astype(int)
    feat = [c for c in edge.columns if c not in EDGE_LABEL]
    rows.append(measure("Edge-IIoTset", "all 61 raw features", edge[feat], y_e))
    s1 = [c for c in feat if c not in EDGE_ID]
    rows.append(measure("Edge-IIoTset", "minus 5 address/time identifiers", edge[s1], y_e))
    s2 = [c for c in s1 if c not in EDGE_NONCE]
    rows.append(measure("Edge-IIoTset", "minus 11 per-connection nonces and ports", edge[s2], y_e))
    s3 = [c for c in s2 if c not in EDGE_CONTENT]
    rows.append(measure("Edge-IIoTset", "minus 7 payload/content columns", edge[s3], y_e))
    Xm = edge_model_space(edge)
    rows.append(measure("Edge-IIoTset", f"project modelling space ({Xm.shape[1]} features, placeholders normalised)",
                        Xm, y_e.loc[Xm.index]))
    per_type = []
    key = pd.util.hash_pandas_object(Xm, index=False)
    for t, idx in edge.loc[Xm.index].groupby("Attack_type").groups.items():
        per_type.append({"attack_type": t, "rows": len(idx), "distinct_vectors": int(key.loc[idx].nunique())})
    pd.DataFrame(per_type).sort_values("rows", ascending=False).to_csv(os.path.join(OUT, "edge_per_type_distinct.csv"), index=False)

    # --- WUSTL-EHMS-2020 -------------------------------------------------------------------------
    w = pd.read_csv(os.path.join(ROOT, "data", "wustl_ehms_2020", "wustl-ehms-2020.csv"), low_memory=False)
    y_w = w["Label"].astype(int)
    wfeat = [c for c in w.columns if c not in ("Label", "Attack Category")]
    rows.append(measure("WUSTL-EHMS-2020", "all 43 raw features", w[wfeat], y_w))
    w_ids = [c for c in ("SrcAddr", "DstAddr", "Sport", "Dport", "SrcMac", "DstMac", "Dir", "Flgs") if c in w.columns]
    ws = [c for c in wfeat if c not in w_ids]
    rows.append(measure("WUSTL-EHMS-2020", f"minus {len(w_ids)} address/port/MAC identifiers", w[ws], y_w))
    bio = [c for c in ("Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate", "Resp_Rate", "ST") if c in w.columns]
    wn = [c for c in ws if c not in bio]
    rows.append(measure("WUSTL-EHMS-2020", "network features only (biometrics removed)", w[wn], y_w))

    # --- CICIDS2017, Friday DDoS day only (partial, reduced 32-column variant) ------------------
    c = pd.read_csv(os.path.join(ROOT, "data", "cicids2017_partial", "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
                    low_memory=False)
    c.columns = [x.strip() for x in c.columns]
    y_c = (c["Label"].astype(str).str.strip() != "BENIGN").astype(int)
    cfeat = [x for x in c.columns if x != "Label"]
    rows.append(measure("CICIDS2017 (Friday DDoS day only, 32-col variant)", "all raw columns", c[cfeat], y_c))
    c_ids = [x for x in cfeat if x in ("Flow ID", "Source IP", "Destination IP", "Source Port", "Timestamp", "Destination Port")]
    cs = [x for x in cfeat if x not in c_ids]
    rows.append(measure("CICIDS2017 (Friday DDoS day only, 32-col variant)", f"minus {len(c_ids)} identifiers", c[cs], y_c))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "dataset_leakage.csv"), index=False)
    iot = {"dataset": "IoTID20", "source": "project measurement, src/preprocessing/clean.py docstring (not re-run here: dataset unreachable from this machine)",
           "raw_rows": 625783, "raw_exact_duplicate_rows": 163946, "raw_duplicate_pct": 100 * 163946 / 625783,
           "model_space_rows": 625415, "model_space_features": 69, "model_space_duplicate_rows": 363884,
           "model_space_duplicate_pct": 100 * 363884 / 625415,
           "corroboration": "Alsulami et al. 2022 report removing 164,087 duplicates of 625,783 rows (26.2%)"}
    json.dump(iot, open(os.path.join(OUT, "iotid20_from_project.json"), "w"), indent=2)
    print(json.dumps(iot))


if __name__ == "__main__":
    main()
