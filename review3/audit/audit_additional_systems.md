# Phase 2.1: Audit of Additional Systems

Scope: the same five checks used on HIDS-IoMT (Berguiga, Harchay, Massaoudi, IEEE Access 13, 2025, pp. 32863 to 32882), applied to two more open access IoMT intrusion detection papers from 2025 and 2026. All numbers below come from full text fetched on 2026-10-20. Locations are given for every number. The re-derivation script is `rederive.py`, its console output is `rederive_output.txt`, and all extracted and recomputed values are in `audit_extracted.json`.

Method caveat: pages were read through WebFetch, which passes the text through an extraction model. The counts and table values were requested verbatim and they reproduce exactly in arithmetic (see below), which is strong evidence that they were transcribed correctly. Even so, check each number against the publisher PDF before submission.

---

## Candidate screening

Ten papers were screened. Two met every requirement: 2024 to 2026, IoMT framing, open access, a numeric confusion matrix readable as text, and metrics reported for the same experiment.

| Paper | Outcome |
|---|---|
| Hosain and Cakmak, Sci Rep 15:22278 (2025), XAI-XGBoost | **Audited (P1)**: counts given in text |
| Tarish et al., Front. AI 9 (2026), Nystrom RFK | **Audited (P2)**: counts in figure alt text |
| Liu, Sci Rep 16:2079 (2026), BiGRU/RBWK, DOI 10.1038/s41598-025-31850-0 | Screened only: partial counts (no TN), no metric table in the fetched text |
| Saeed et al., Sci Rep 16:2097 (2025), DOI 10.1038/s41598-025-31897-z | Rejected: matrices only as images |
| Shaikh et al., Front. Med. 12:1524286 (2025), DOI 10.3389/fmed.2025.1524286 | Rejected: matrices only as images |
| Alsolami et al., Sensors 24:5937 (2024), DOI 10.3390/s24185937 | Rejected: matrices only as figures |
| Balhareth et al., Sensors 26:2501 (2026), DOI 10.3390/s26082501 | Rejected: no numeric matrix in the fetched text |
| Alnasrallah et al., PLOS ONE 20(7) (2025), DOI 10.1371/journal.pone.0327137 | Rejected: matrices only as images |
| Alsolami and Ilyas, Sensors 26:1592 (2026), DOI 10.3390/s26051592 | Rejected: results section truncated, matrices as figures |
| Ammar and Cherif, IoT 6(4):74 (2025), DOI 10.3390/iot6040074 | Rejected: no matrix counts |

Pages that could not be fetched:
* nature.com article pages (client error). Both Sci Rep papers were read through PMC instead.
* OpenAlex and Europe PMC search APIs (HTTP 429).
* Three PMC articles that returned a reCAPTCHA page: PMC13363931, PMC13503730, PMC13402581.

---

## P1. Hosain and Cakmak 2025 (XAI-XGBoost)

**Citation.** Y. Hosain, M. Cakmak, "XAI-XGBoost: an innovative explainable intrusion detection approach for securing internet of medical things systems", *Scientific Reports* 15, 22278 (2025).
* DOI: 10.1038/s41598-025-07790-0
* URL read: https://pmc.ncbi.nlm.nih.gov/articles/PMC12218340/ (full text including the tail)
* Dataset: WUSTL-EHMS-2020, binary classification.

### Extracted values

| Item | Value | Location |
|---|---|---|
| Confusion matrix | TN 2836 (normal correct), TP 2262 (attack correct), FP 38 ("normal instances as attacks (false positives)"), FN 2 ("attack instances as normal (false negatives)") | Section "Confusion matrix analysis", text. Fig. 5 is an image. |
| Positive class | Attack. TP is defined as "attack instances correctly detected as attacks" | Section 6, evaluation metrics |
| Reported metrics | Acc 99.22, Prec 98.35, Rec 99.91, F1 99.12 (%) | Table 5 (proposed model row), Table 6, Abstract |
| Per-class metrics | Normal(0): P 100, R 99, F1 99, support 2874. Attack(1): P 98, R 100, F1 99, support 2264. Accuracy 99.22, total 5138 | Table 4 |
| Formulae | Eqs. 10 to 13 are images and could not be read. The text definitions of precision and recall are the standard attack-positive ones. | Section 6 |
| Original size | 14,272 normal, 2,046 attack, 16,318 total | Table 1 |
| After balancing | "adjusted to have 14,271 normal samples and 11,417 attack samples". Balancing factor b = 0.8. | Section "Data balancing approach" |
| Balancing method | Hybrid: oversampling of the minority class plus undersampling of the majority class | Section "Data balancing approach" |
| Split | "split into 80% for training and 20% for testing". Train count not stated. Test count 5138 from the Table 4 support column. | Section "Data preprocessing" |
| Order of balancing and split | Not stated in the text. The preprocessing list places "Dataset splitting" before the separate balancing section. | Data preprocessing section |
| Duplicates | Not mentioned. A search for "duplicat" over the full text, including the tail, found nothing. | Whole text |
| Thresholds | No decision threshold value is given. Thresholds are mentioned only in the ROC description. | Section 6 |

### Re-derivation (from TP 2262, TN 2836, FP 38, FN 2; n = 5138)

| Metric | Printed | Attack positive | Normal positive | Swapped (attack positive) |
|---|---|---|---|---|
| Accuracy | 99.22 | 99.2215 | 99.2215 | n/a |
| Precision | 98.35 | **98.3478** | 99.9295 | 99.9117 |
| Recall | 99.91 | **99.9117** | 98.6778 | 98.3478 |
| F1 | 99.12 | **99.1236** | 99.2997 | n/a |
| Macro P / R / F1 | Table 4: 99 / 99 / 99 | 99.1387 / 99.2947 / 99.2116 | | |
| Weighted P / R / F1 | Table 4: 99 / 99 / 99 | 99.2326 / 99.2215 / 99.2221 | | |

* The matrix row totals (2264 attack, 2874 normal) equal the Table 4 supports.
* The Table 4 per-class integers are the rounded recomputed values: Normal P 99.93 rounds to 100 and R 98.68 to 99; Attack P 98.35 rounds to 98 and R 99.91 to 100.

### Split arithmetic

* Balanced total: 14,271 + 11,417 = 25,688. 20% of 25,688 is 5,137.6, which rounds to **5,138, the exact test support**.
* 20% of the original 16,318 would be 3,263.6.
* The test set holds **2,264 attack samples**, but the entire original dataset has only **2,046** attack records (Table 1). At least 218 test attack samples must therefore be synthetic or duplicated. Oversampling was applied before the train/test partition.
* 11,417 is 0.8 × 14,271 (11,416.8) rounded, which is consistent with b = 0.8 applied to the full dataset.

### Verdicts

| Check | Verdict | Evidence |
|---|---|---|
| a. Positive-class inversion | NOT PRESENT | Printed P/R/F1 match attack-positive to 2 dp. Attack is defined as TP. |
| b. Transposed precision/recall | NOT PRESENT | Printed precision 98.35 equals recomputed precision, not recall (99.91). The formula images could not be read, but the text definitions are correct. |
| c. Post-oversampling partitioning | **CONFIRMED** | The test set equals 20% of the balanced 25,688, and its 2,264 test attacks exceed the 2,046 attacks in the whole dataset. |
| d. Duplicates not handled | **CONFIRMED (not mentioned)** | No "duplicat" anywhere in the text. |
| e. Other arithmetic inconsistency | NOT PRESENT | Accuracy, P, R and F1 all reproduce to 2 dp. Per-class and support values are consistent. |

Interpretation: the metric arithmetic is correct, but the reported 99.22% is measured on a test set that contains synthetic minority samples. The same leakage pattern found in HIDS-IoMT (check c) recurs here. The evidence is even more direct: the test set holds more attack rows than exist in the source data.

---

## P2. Tarish et al. 2026 (leakage-free Nystrom RFK)

**Citation.** H. A. Tarish, R. Hassan, K. A. Zainol Ariffin, M. M. Jaber, "Leakage-free evaluation of task-aware Nystrom sampling for IoMT intrusion detection", *Frontiers in Artificial Intelligence* 9 (2026), published 28 September 2026.
* DOI: 10.3389/frai.2026.1957727
* URL read: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1957727/full
* Datasets: ECU-IoHT and WUSTL-EHMS-2020, binary classification.

### Extracted values

| Item | Value | Location |
|---|---|---|
| Matrix, ECU-IoHT, RFK (a priori) | TN 3,516; FP 2; FN 1,120; TP 12,044. Actual values are on the vertical axis and predicted values on the horizontal. | Figure 1 alt text. The Fig. 2 caption names this matrix. |
| Matrix, WUSTL, RFK (a priori) | normal→normal 2,130; normal→attack 11; attack→normal 163; attack→attack 144 | Figure 3 alt text. The Fig. 4 caption names this matrix. |
| Positive class | Attack: "for the attack (positive) class specifically" | Section 3.1 |
| Reported, ECU | Acc 93.27, Prec 99.98, Rec 91.49, F1 95.55, AUC 0.9922, n = 16,682 | Table 4, RFK (a priori) row |
| Reported, WUSTL | Acc 92.89, Prec 92.90, Rec 46.91, F1 62.34, AUC 0.8817, n = 2,448 (2,141 normal, 307 attack) | Table 5, RFK (a priori) row; Section 4.2 |
| Formulae | No accuracy, precision, recall or F1 formulae are printed | Whole text |
| Split | 70/15/15 stratified, random_state = 42, "applied to the raw feature matrix before any feature selection" | Section 3.2 |
| Dataset totals and train/val counts | Not stated | n/a |
| Oversampling | Main results use class weighting without resampling. SMOTE appears only in Table 7, "applied to the transformed training features only". | Sections 3.1, 4.2; Table 7 |
| Duplicates | Not mentioned. "duplicat" and "redundan" are absent. | Whole text |
| Thresholds | ECU: decision_function = 0. WUSTL: p = 0.5. | Section 3.7 |

### Re-derivation

| Metric | ECU printed | ECU attack positive | ECU normal positive | WUSTL printed | WUSTL attack positive | WUSTL normal positive |
|---|---|---|---|---|---|---|
| Accuracy | 93.27 | 93.2742 | | 92.89 | 92.8922 | |
| Precision | 99.98 | **99.9834** | 75.8412 | 92.90 | **92.9032** | 92.8914 |
| Recall | 91.49 | **91.4919** | 99.9431 | 46.91 | **46.9055** | 99.4862 |
| F1 | 95.55 | **95.5494** | 86.2399 | 62.34 | **62.3377** | 96.0758 |
| Macro P / R / F1 | n/a | 87.91 / 95.72 / 90.89 | | n/a | 92.90 / 73.20 / 79.21 | |
| Weighted P / R / F1 | n/a | 94.89 / 93.27 / 93.59 | | n/a | 92.89 / 92.89 / 91.84 | |

* The matrix totals match the stated test sizes: ECU 16,682 and WUSTL 2,448 (2,141 normal, 307 attack).
* The WUSTL test set is 15% of 16,318, which is 2,447.7.
* The attack share is 12.54% in the test set and 12.54% in the original data. This is consistent with a stratified split of untouched data.

### Verdicts

| Check | Verdict | Evidence |
|---|---|---|
| a. Positive-class inversion | NOT PRESENT | Attack-positive values reproduce to 2 dp on both datasets. Normal-positive values do not. |
| b. Transposed precision/recall | NOT PRESENT | ECU: printed P 99.98 equals recomputed precision (recall is 91.49). No formulae are printed. |
| c. Post-oversampling partitioning | NOT PRESENT | The split comes first (Sec. 3.2). SMOTE is applied to training data only (Table 7). Test sizes and class shares match the raw data. |
| d. Duplicates not handled | **CONFIRMED (not mentioned)** | No mention of duplicate removal. |
| e. Other arithmetic inconsistency | NOT PRESENT (arithmetic). Minor presentation issue. | All metrics reproduce. However, Fig. 1 and Fig. 3 carry confusion-matrix alt text under ANOVA and convergence captions: the alt text and captions are misaligned. |

Interpretation: this paper is clean on checks a, b, c and e. It is a useful contrast case. Under a leakage-free protocol, attack recall on WUSTL-EHMS-2020 is 46.91% (Table 5), and the best F1 among all baselines is 68.14 (MLP). That is far below the 99%+ figures reported by papers that oversample before splitting. Caveat: the counts come from image alt text, not a printed table, so check them against the rendered figure.

---

## Additional observation (screened paper, not fully audited)

Liu, Sci Rep 16:2079 (2026), DOI 10.1038/s41598-025-31850-0, PMC12808284:
* Section "Confusion matrix analysis" states that the model "detects correctly normal samples with 13,850 true positives". This calls correctly classified Normal records true positives, which is the same Normal-as-positive framing seen in HIDS-IoMT.
* The FP (186) and FN (422) values given alongside it, with 13,850, sum to 14,458. That exceeds the roughly 2,448 records a 15% test split of 16,318 would hold. The text also gives two different split ratios: 85:15 in the text and 80/20 in Table 3.
* No TN count and no metric table were visible in the fetched text, so checks a to e could not be completed. **CANNOT DETERMINE.**

---

## Cross-paper comparison

| Check | HIDS-IoMT (Berguiga 2025, IEEE Access) | P1 XAI-XGBoost (Hosain 2025, Sci Rep) | P2 Nystrom RFK (Tarish 2026, Front. AI) |
|---|---|---|---|
| a. Positive-class inversion | CONFIRMED | NOT PRESENT | NOT PRESENT |
| b. Transposed precision/recall | CONFIRMED | NOT PRESENT | NOT PRESENT |
| c. Post-oversampling partitioning | CONFIRMED | **CONFIRMED** (test = 20% of balanced 25,688; 2,264 test attacks > 2,046 real attacks) | NOT PRESENT (split first, SMOTE on training only) |
| d. Duplicates not handled | CONFIRMED (not mentioned) | CONFIRMED (not mentioned) | CONFIRMED (not mentioned) |
| e. Other arithmetic inconsistency | None beyond a and b | NOT PRESENT | NOT PRESENT (figure caption and alt text misaligned) |

### Summary

* **Duplicates (d)** are unaddressed in all three papers.
* **Oversampling before the split (c)** recurs in the second paper, and its evidence is arithmetic, independent of the authors' wording.
* **The metric-definition errors (a, b)** did not recur in either audited paper. Both reproduce exactly under attack-as-positive. These errors appear specific to HIDS-IoMT, although the Liu 2026 text shows the same Normal-as-TP wording.
* **P2** is the one paper built around a leakage-free protocol, and it reports much lower minority-class recall on WUSTL-EHMS-2020. This supports the hypothesis that the near-perfect scores elsewhere are inflated by the evaluation protocol.
