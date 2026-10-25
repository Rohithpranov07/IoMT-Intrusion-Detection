"""Phase 2.1 audit: re-derive metrics from confusion-matrix counts printed in two
additional IoMT IDS papers. Counts are copied from fetched full text (see
audit_extracted.json for source locations). Pure arithmetic, no heavy compute."""
import json

PAPERS = {
    "P1_Hosain2025_XAI-XGBoost_WUSTL": {
        # Sci Rep 15:22278, section "Confusion matrix analysis" (text; Fig. 5 is image)
        "TP": 2262, "TN": 2836, "FP": 38, "FN": 2,
        "printed": {"accuracy": 99.22, "precision": 98.35, "recall": 99.91, "f1": 99.12},
        "printed_src": "Table 5 (proposed model row), Table 6, Abstract",
        "printed_per_class": {  # Table 4, integer percentages
            "Normal": {"precision": 100, "recall": 99, "f1": 99, "support": 2874},
            "Attack": {"precision": 98, "recall": 100, "f1": 99, "support": 2264},
        },
    },
    "P2_Tarish2026_RFK_ECU-IoHT": {
        # Front. AI 9:1957727, Figure 1 alt text (actual rows, predicted cols)
        "TP": 12044, "TN": 3516, "FP": 2, "FN": 1120,
        "printed": {"accuracy": 93.27, "precision": 99.98, "recall": 91.49, "f1": 95.55},
        "printed_src": "Table 4, RFK (a priori) row; attack-class Prec/Rec/F1",
    },
    "P2_Tarish2026_RFK_WUSTL": {
        # Front. AI 9:1957727, Figure 3 alt text
        "TP": 144, "TN": 2130, "FP": 11, "FN": 163,
        "printed": {"accuracy": 92.89, "precision": 92.90, "recall": 46.91, "f1": 62.34},
        "printed_src": "Table 5, RFK (a priori) row; attack-class Prec/Rec/F1",
    },
}

def prf(tp, fp, fn):
    p = tp / (tp + fp); r = tp / (tp + fn)
    return p, r, 2 * p * r / (p + r)

out = {}
for name, d in PAPERS.items():
    TP, TN, FP, FN = d["TP"], d["TN"], d["FP"], d["FN"]
    n = TP + TN + FP + FN
    acc = (TP + TN) / n
    pa, ra, fa = prf(TP, FP, FN)          # Attack as positive
    pn, rn, fn_ = prf(TN, FN, FP)         # Normal as positive
    n_att, n_norm = TP + FN, TN + FP
    w = lambda a, b: (a * n_att + b * n_norm) / n
    res = {
        "n_total": n, "n_attack_actual": n_att, "n_normal_actual": n_norm,
        "accuracy": acc,
        "attack_pos": {"precision": pa, "recall": ra, "f1": fa},
        "normal_pos": {"precision": pn, "recall": rn, "f1": fn_},
        "transposed_attack_pos": {"precision_if_swapped": ra, "recall_if_swapped": pa},
        "macro": {"precision": (pa + pn) / 2, "recall": (ra + rn) / 2, "f1": (fa + fn_) / 2},
        "weighted": {"precision": w(pa, pn), "recall": w(ra, rn), "f1": w(fa, fn_)},
    }
    pr = d["printed"]
    res["match_2dp_pct"] = {
        "accuracy": round(acc * 100, 2) == pr["accuracy"],
        "attack_pos_precision": round(pa * 100, 2) == pr["precision"],
        "attack_pos_recall": round(ra * 100, 2) == pr["recall"],
        "attack_pos_f1": round(fa * 100, 2) == pr["f1"],
        "normal_pos_precision": round(pn * 100, 2) == pr["precision"],
        "normal_pos_recall": round(rn * 100, 2) == pr["recall"],
        "swapped_precision": round(ra * 100, 2) == pr["precision"],
        "swapped_recall": round(pa * 100, 2) == pr["recall"],
    }
    out[name] = res
    print(f"\n== {name}  (printed: {d['printed_src']})")
    print(f"n={n} attack={n_att} normal={n_norm}")
    print(f"accuracy        {acc*100:9.4f}  printed {pr['accuracy']}")
    for lab, (p, r, f) in [("Attack+", (pa, ra, fa)), ("Normal+", (pn, rn, fn_))]:
        print(f"{lab} P {p*100:8.4f} R {r*100:8.4f} F1 {f*100:8.4f}   printed P {pr['precision']} R {pr['recall']} F1 {pr['f1']}")
    print("macro   P {:.4f} R {:.4f} F1 {:.4f}".format(*(v*100 for v in res['macro'].values())))
    print("weighted P {:.4f} R {:.4f} F1 {:.4f}".format(*(v*100 for v in res['weighted'].values())))
    print("matches:", res["match_2dp_pct"])

# Split arithmetic checks
print("\n== Split arithmetic")
s = {}
# P1: Table 1 original counts, balancing text counts, Table 4 test support
orig_n, orig_norm, orig_att = 16318, 14272, 2046
bal_norm, bal_att = 14271, 11417
test_support = 2874 + 2264
s["P1_balanced_total"] = bal_norm + bal_att
s["P1_20pct_of_balanced"] = 0.2 * (bal_norm + bal_att)
s["P1_20pct_of_original"] = 0.2 * orig_n
s["P1_test_support"] = test_support
s["P1_test_attack_exceeds_all_original_attacks"] = 2264 > orig_att
s["P1_balancing_factor_check_0.8x14271"] = 0.8 * bal_norm
# P2: WUSTL original 16,318 (paper P1 Table 1); test reported 2,448 at 15%
s["P2_WUSTL_15pct_of_original"] = 0.15 * orig_n
s["P2_WUSTL_attack_share_test"] = 307 / 2448
s["P2_WUSTL_attack_share_original"] = orig_att / orig_n
for k, v in s.items():
    print(f"{k}: {v}")
out["split_checks"] = s

json.dump(out, open("/home/claude/r3/audit/rederived_values.json", "w"), indent=2)
