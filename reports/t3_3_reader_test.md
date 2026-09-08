# Reader test — can a non-specialist use these alerts? (T3.3 VERIFY)

**For the reader:** you do not need any machine-learning background, and you should not
look at the code before answering. That is the point of the exercise — if these alerts
only make sense to someone who built the model, they have failed at their job.

## Background you need (and no more)

A monitoring system watches network traffic going to medical devices on a hospital network.
It reads the traffic in short windows of 10 consecutive records. For each window it decides
**Attack** or **Normal**, and when it says Attack it produces the note below explaining
itself. Three separate detectors vote on each decision, and the note says how much weight
each one carried.

Each note begins with a status line in square brackets. Part of this exercise is finding
out whether that line means anything to you without being told what it means.

Each measurement is described in words, with the underlying technical field name in
brackets after it. You should not need the bracketed name; it is there so a network
engineer could trace the alert back to the raw traffic.

## Your task

For **each** alert below, write down — in your own words, in one or two sentences:

1. **What did the system decide, and how sure was it?**
2. **Why did it decide that?** Name the specific measurements it based the decision on.
3. **Would you act on this alert?** Why or why not?

Then answer once, overall:

4. **Did any alert below look less reliable than the others?** If so, which, and what told
   you?
5. **What did you not understand?** Every term that stopped you is a defect worth fixing.
6. **Were the measurement descriptions clear**, or did you find yourself relying on the
   bracketed technical names? If any description left you guessing, quote it.

---

## Alert 1

```
[VERIFIED] Removing the top 1 cited measurement changes this decision by 98%, against 19% for unrelated ones.

ALERT: this traffic window was classified as ATTACK with 98% confidence.

What pointed to Attack:
  1. the average size of packets in the conversation (Pkt_Size_Avg)
       22% of the total evidence, strongest at record 9 of 10.
  5. the size of the smallest packet in the conversation (Pkt_Len_Min)
       6% of the total evidence, strongest at record 9 of 10.

What argued against it (these looked more like Normal):
  2. the average amount of useful data per packet the device sent back (Bwd_Seg_Size_Avg)
       13% of the total evidence, strongest at record 9 of 10.
  3. the average packet size in the conversation (Pkt_Len_Mean)
       8% of the total evidence, strongest at record 10 of 10.
  4. how many packets were requests to open a new connection (a flood of these is how port scans and SYN attacks look) (SYN_Flag_Cnt)
       6% of the total evidence, strongest at record 10 of 10.

Detector agreement: the cnn detector carried the most weight (34%); weights across all detectors were cnn 34%, bilstm 33%, transformer 33%.
(Explanation produced by SHAP in 0.18s. 'Attack' is the positive class.)
```

**Your answers**

1. Decision and confidence: 
2. Reason: 
3. Would you act on it? 

---

## Alert 2

```
[VERIFIED] Removing the top 1 cited measurement changes this decision by 98%, against 20% for unrelated ones.

ALERT: this traffic window was classified as ATTACK with 100% confidence.

What pointed to Attack:
  1. the average size of packets in the conversation (Pkt_Size_Avg)
       16% of the total evidence, strongest at record 7 of 10.
  4. the largest packet sent to the device (Fwd_Pkt_Len_Max)
       5% of the total evidence, strongest at record 9 of 10.

What argued against it (these looked more like Normal):
  2. how many packets were requests to open a new connection (a flood of these is how port scans and SYN attacks look) (SYN_Flag_Cnt)
       14% of the total evidence, strongest at record 2 of 10.
  3. the average amount of useful data per packet the device sent back (Bwd_Seg_Size_Avg)
       6% of the total evidence, strongest at record 9 of 10.
  5. the smallest packet sent to the device (Fwd_Pkt_Len_Min)
       4% of the total evidence, strongest at record 9 of 10.

Detector agreement: the cnn detector carried the most weight (33%); weights across all detectors were cnn 33%, bilstm 33%, transformer 33%.
(Explanation produced by SHAP in 0.18s. 'Attack' is the positive class.)
```

**Your answers**

1. Decision and confidence: 
2. Reason: 
3. Would you act on it? 

---

## Alert 3

```
[UNVERIFIED] removing the cited measurements barely changes this decision (best gain -13.4% at k=5, threshold 5%) — the model's decision here is spread across many measurements, so no short list explains it

ALERT: this traffic window was classified as ATTACK with 68% confidence.

What pointed to Attack:
  3. the average amount of useful data per packet the device sent back (Bwd_Seg_Size_Avg)
       7% of the total evidence, strongest at record 5 of 10.
  4. the average packet size in the conversation (Pkt_Len_Mean)
       6% of the total evidence, strongest at record 7 of 10.

What argued against it (these looked more like Normal):
  1. how many packets were requests to open a new connection (a flood of these is how port scans and SYN attacks look) (SYN_Flag_Cnt)
       13% of the total evidence, strongest at record 2 of 10.
  2. the average size of packets in the conversation (Pkt_Size_Avg)
       12% of the total evidence, strongest at record 9 of 10.
  5. which network protocol was used (TCP, UDP, or other) (Protocol)
       6% of the total evidence, strongest at record 1 of 10.

Detector agreement: the bilstm detector carried the most weight (39%); weights across all detectors were cnn 32%, bilstm 39%, transformer 29%.
(Explanation produced by SHAP in 1.73s. 'Attack' is the positive class.)
```

**Your answers**

1. Decision and confidence: 
2. Reason: 
3. Would you act on it? 

---

## Overall

4. Did any alert look less reliable? Which, and what told you? 

5. What did you not understand? 

---

# Scoring guidance — for the team, AFTER the reader has answered

> Do not show this section to the reader beforehand.

### What counts as a pass

The gate (`TRD.md §9`, T3.3's VERIFY) is that the reader **correctly states why the flow
was flagged**. Concretely, a pass means their answer to question 2 names at least one of
the measurements the alert actually cited, and does not invent a reason the alert did not
give. They do **not** need to know what the measurement means in networking terms —
`Pkt_Size_Avg` naming a packet-size average is enough. Judgement about whether that
*should* be suspicious is a networking question, not an explainability one.

### The answer that matters most

**Question 4.** One alert below is one the system itself does not stand behind: its
`[UNVERIFIED]` line says removing the measurements it cites barely changes the decision, so
the stated reason is not supported by the model. If the reader did not notice, **the
labelling has failed**, and that is a more serious finding than any wording problem —
it means an operator could act on a rationale the system knows to be unsupported.

The fix in that case is presentational (make the banner louder, or withhold unverified
explanations from the default view), not a change to the model. Record the outcome either
way in `reports/t3_3_xai_evaluation.md`.

### Two changes already made in response to this gate

Both were defects a reader test would have surfaced, found before the test was run, and
fixed rather than left for the reader to trip over:

1. **Measurements are now described in words.** Alerts previously cited raw column names
   such as `Init_Bwd_Win_Byts` and `Bwd_Seg_Size_Avg`. Those are *traceable* to the data
   but not *comprehensible* to the intended reader, and the two had been conflated.
   `src/xai/feature_glossary.py` now covers 100% of the 62 features the model can cite.
2. **Evidence is split by direction.** Features are ranked by absolute influence, which is
   correct and is what makes the shares sum to 100% — but it meant the top-ranked
   measurement sometimes argued *against* its own verdict (12% of SHAP explanations, 88%
   of LIME's). Alerts now separate 'what pointed to Attack' from 'what argued against it',
   keeping every feature and its true direction while removing the contradiction.

Question 6 exists to check whether the first change actually worked. If the reader still
leans on the bracketed technical names, the descriptions are not doing their job.

### Recording the result

| | |
|---|---|
| Reader (name, role) | |
| Date | |
| Q2 correct for each alert? | |
| Q4 — spotted the unverified alert? | |
| Terms that blocked them (Q5) | |
| **VERIFY passed?** | |

Paste the completed table into `reports/t3_3_xai_evaluation.md` under a new
'Reader test' heading. Until then, T3.3's VERIFY block is **incomplete** — the quantitative
metrics in that report stand on their own, but they are not a substitute for this gate.
