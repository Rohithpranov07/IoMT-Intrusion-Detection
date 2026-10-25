# Reader protocol: can a non-specialist use the alert explanations?

Phase 6 of the Review 3 roadmap. Study materials: `study.html` (published app), `export_study_alerts.py`, `analyse_reader_study.py`, this protocol and the consent form below.

## 1. Questions and hypotheses

- **RQ1.** Can a reader with no machine-learning background state correctly why a flow was flagged?
- **RQ2.** Does certification change how readers treat an alert?
- **RQ3.** Do contradictory explanations (top-ranked evidence argues against the verdict) mislead readers?

Pre-registered hypotheses, written before any session is run:

- **H1.** Reason-stated-correctly rate is higher for CERTIFIED than for UNCERTIFIED alerts.
- **H2.** Trust (q5) is lower for UNCERTIFIED than for CERTIFIED alerts, i.e. readers notice the banner.
- **H3.** CONTRADICTORY alerts produce more selections of counter-evidence as the reason than CERTIFIED alerts.
- **H4 (exploratory).** Time-to-decision is longer for UNCERTIFIED and CONTRADICTORY alerts.

A null result on H2 is the most important possible finding: it would mean an operator acts on a rationale the system itself does not stand behind.

## 2. Design

Within-participant. Every participant sees all 20 alerts: 8 CERTIFIED, 6 UNCERTIFIED, 6 CONTRADICTORY, all real outputs of the trained ensemble produced by `export_study_alerts.py`. The order is shuffled per participant from the participant code (seeded, reproducible). Two distractor measurements that the alert never cites are added to each Q2 checklist.

## 3. Participants

10 to 15 adults: students, researchers or clinicians, none of whom has worked on this project's model. Exclude team members and anyone who has read the project's code or reports. Record only a participant code.

## 4. Procedure (about 25 minutes, one participant at a time)

1. Facilitator opens the study app, loads `alerts_20.json`, enters the next code (P01, P02, ...).
2. Participant reads the information screen and ticks consent. A paper copy of the consent form (below) is signed and kept by the facilitator.
3. Four background questions.
4. 20 alerts. For each: verdict, system confidence band, reason (checklist plus one sentence), action, self-confidence, trust. Timing is automatic.
5. Post-questionnaire: which alerts looked less reliable, what was unclear, three agreement items.
6. Facilitator returns to the console. After the last participant: **Export CSV** and **Export JSON**.

The facilitator does not explain any term, banner or number during a session. Answer "please use your best judgement".

## 5. Measures and scoring

| Measure | Definition |
|---|---|
| Reason correct (primary) | at least one measurement the alert cited FOR its verdict, and no counter-evidence or distractor selected (the pass rule already fixed in `make_reader_test.py`) |
| Verdict read | q1 equals the alert's verdict |
| Confidence band read | q1c equals the band containing the printed confidence |
| Action | act / escalate / ignore |
| Self-confidence, trust | 1 to 7 |
| Time-to-decision | display to submit, ms |
| Spotted unreliable | free text, coded by two team members independently; report Cohen's kappa |

## 6. Analysis

`python3 reader_study/analyse_reader_study.py reader_study_responses.csv` prints the results table and tests. Participant-level means per category; paired Wilcoxon signed-rank tests; Bonferroni correction across every test run; chi-square for action by category. With 10 to 15 participants the study is a pilot: report effect sizes and medians alongside p-values and do not generalise to clinicians.

## 7. Ethics

Minimal risk: no personal data beyond a code and four background answers, no deception beyond not explaining the banner in advance (disclosed at debrief). Ask the course faculty whether the department requires Institutional Ethics Committee review for a minimal-risk classroom study and follow that answer before the first session. Data stays on the facilitator laptop until exported, then in the team's private project folder.

## 8. Debrief (read aloud after the post-questionnaire)

"Some of the notes you read were ones the system itself marks as unreliable, shown by the status line at the top. We did not tell you that in advance because we wanted to see whether the line is noticed on its own. Thank you."

---

# Consent form

**Study:** Understanding automated security alerts on a hospital network
**Team:** Malika Balaji, Nehaa Prabhu, Rohith Pranov, Sanhit Muthuswamy, Rakshan V P, School of Computer Science and Engineering, Vellore Institute of Technology

- I have read the information sheet and had the chance to ask questions.
- I understand that taking part is voluntary and that I can stop at any time without giving a reason, in which case my answers are discarded.
- I understand that my answers are stored under a code, not my name, and will be reported only as group statistics in a course report and a possible research paper.
- I confirm that I am 18 or older.
- I agree to take part.

Participant code: ________    Signature: ____________________    Date: __________

Facilitator: ____________________    Signature: ____________________
