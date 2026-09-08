"""Held-out attack-type experiment for incremental learning (T3.5; TRD.md §5.2, §9).

T3.5 requires "a before/after evaluation: measure detection performance on prior attack types
immediately before and after an incremental update on a new attack type", and `TRD.md §9`'s gate is
that prior-class detection must not regress.

That cannot be measured on the T2.6 ensemble, which was trained on every attack type IoTID20
contains — there is no novel type left to introduce. So this script runs the whole experiment
end to end:

  1. Hold out one attack sub-type entirely (`MITM ARP Spoofing` by default).
  2. Train a fresh ensemble on the remaining types, using the same leakage-free, session-split
     pipeline as T2.6 so the result is comparable.
  3. Measure detection on the held-out type. It should be poor — that is what makes it novel.
  4. Apply an incremental update from a small sample of the new type.
  5. Re-measure BOTH the new type and the prior types.

Two mechanisms are compared: the replay-buffer fine-tuning this project adopts, and the
fusion-priors-only approach `TRD.md §5.2` lists first and `src/adaptive/incremental.py` rejects.
The rejection is therefore measured, not asserted.

Usage:
    .venv/bin/python scripts/evaluate_incremental_learning.py
"""

from __future__ import annotations

import copy
import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.adaptive.incremental import (  # noqa: E402
    BUFFER_CAPACITY,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LEARNING_RATE,
    REPLAY_RATIO,
    ReplayBuffer,
    run_fusion_only_update,
    run_incremental_update,
)
from src.config import RANDOM_STATE, REPORTS_DIR  # noqa: E402
from src.evaluation.metrics import POSITIVE_CLASS_STATEMENT  # noqa: E402
from src.models import bilstm_branch, cnn_branch, transformer_branch  # noqa: E402
from src.models.train_utils import (  # noqa: E402
    compute_class_weights,
    scale_sequences,
    set_global_seeds,
    train_branch,
)
from src.preprocessing.clean import clean_iotid20, load_iotid20  # noqa: E402
from src.preprocessing.feature_select import (  # noqa: E402
    IOTID20_N_FEATURES,
    RandomForestFeatureSelector,
)
from src.preprocessing.sequence_builder import (  # noqa: E402
    SEQUENCE_LENGTH,
    TRAIN_STRIDE,
    assign_sessions,
    build_sequences,
    split_sessions,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s")

#: The attack sub-type held out of training and introduced incrementally. MITM ARP Spoofing is
#: chosen because it is behaviourally unlike the Mirai/DoS flooding families that dominate
#: IoTID20 — an easy hold-out would make the experiment meaningless, since the model would already
#: detect it and there would be nothing to learn.
NOVEL_ATTACK_TYPE: str = "MITM ARP Spoofing"

#: Labelled windows of the new type supplied to the update. Deliberately small: the premise of
#: incremental learning is that a full retrain's worth of data is NOT available.
N_NEW_SAMPLES: int = 300


def main() -> None:
    """Run the hold-out, train, update, and re-measure experiment."""
    set_global_seeds(RANDOM_STATE)

    raw = load_iotid20()
    X, y, meta = clean_iotid20(raw, drop_duplicates=True)
    sessions = assign_sessions(meta)

    dataset = build_sequences(
        X, y, meta,
        sequence_length=SEQUENCE_LENGTH,
        stride=TRAIN_STRIDE,
        session_ids=sessions,
        aux_labels=meta["Sub_Cat"].to_numpy(),
    )
    types = np.asarray(dataset.aux_labels, dtype=object)
    print(f"Windows: {len(dataset.X):,}")
    print("Attack sub-types present:")
    for name, count in zip(*np.unique(types, return_counts=True)):
        marker = "  <-- HELD OUT" if name == NOVEL_ATTACK_TYPE else ""
        print(f"  {str(name):<26} {count:>7,}{marker}")

    if NOVEL_ATTACK_TYPE not in set(types):
        raise SystemExit(f"{NOVEL_ATTACK_TYPE!r} not present in the windowed data")

    # Session-level split first, so windows never leak between folds (architecture_decision §1.4).
    train_sessions, test_sessions, val_sessions = split_sessions(dataset.session_ids, dataset.y)
    is_train = np.isin(dataset.session_ids, train_sessions)
    is_test = np.isin(dataset.session_ids, test_sessions)
    is_val = np.isin(dataset.session_ids, val_sessions)
    is_novel = types == NOVEL_ATTACK_TYPE

    # Feature selection on training rows of the KNOWN types only.
    row_train = sessions.isin(train_sessions).to_numpy() & (
        meta["Sub_Cat"].to_numpy() != NOVEL_ATTACK_TYPE
    )
    selector = RandomForestFeatureSelector(
        n_features=IOTID20_N_FEATURES, random_state=RANDOM_STATE
    )
    selector.fit(X[row_train], y[row_train])
    keep = [dataset.feature_names.index(f) for f in selector.selected_features_]
    windows = dataset.X[:, :, keep]

    # Stage 1: train WITHOUT the novel type.
    base_train = is_train & ~is_novel
    base_val = is_val & ~is_novel
    X_train, y_train = windows[base_train], dataset.y[base_train]
    X_val, y_val = windows[base_val], dataset.y[base_val]
    X_test, y_test, test_types = windows[is_test], dataset.y[is_test], types[is_test]

    X_train_s, X_val_s, X_test_s = scale_sequences(X_train, X_val, X_test)
    print(f"\nTraining on {len(X_train_s):,} windows excluding {NOVEL_ATTACK_TYPE!r}; "
          f"validating on {len(X_val_s):,}; evaluating on {len(X_test_s):,} test windows "
          f"({int((test_types == NOVEL_ATTACK_TYPE).sum()):,} of the novel type)")

    class_weights = compute_class_weights(y_train)
    models = {}
    for name, builder in (
        ("cnn", cnn_branch.build_cnn_branch),
        ("bilstm", bilstm_branch.build_bilstm_branch),
        ("transformer", transformer_branch.build_transformer_branch),
    ):
        set_global_seeds(RANDOM_STATE)
        model = builder(SEQUENCE_LENGTH, X_train_s.shape[2])
        # Early stopping uses the SESSION-LEVEL validation fold. An earlier version sliced the
        # last 10% of the training array instead; because windows are session-ordered, that slice
        # was often a single session of one class, and early stopping on validation F1 then
        # selected a degenerate all-Attack model (false-positive rate 0.94).
        train_branch(
            model, X_train_s, y_train, X_val_s, y_val,
            class_weights=class_weights, verbose=0,
        )
        models[name] = model
        accuracy = float((model.predict(X_test_s, verbose=0).argmax(1) == y_test).mean())
        print(f"  trained {name} (test accuracy {accuracy:.4f})")

    # Stage 2: the incremental update sample, drawn from TRAINING sessions only.
    rng = np.random.default_rng(RANDOM_STATE)
    novel_pool = np.flatnonzero(is_train & is_novel)
    if len(novel_pool) == 0:
        raise SystemExit("no novel-type windows fell in the training sessions")
    picked = rng.choice(novel_pool, size=min(N_NEW_SAMPLES, len(novel_pool)), replace=False)
    X_new = scale_sequences(X_train, windows[picked])[1]
    y_new = dataset.y[picked]

    # Replay buffer filled from the prior classes the model already knows.
    buffer = ReplayBuffer(capacity=BUFFER_CAPACITY, random_state=RANDOM_STATE)
    buffer.add(X_train_s, y_train, types[base_train])

    print(f"\nIncremental update: {len(X_new):,} new windows, "
          f"replay buffer holds {len(buffer):,} across {len(buffer.classes_)} classes")

    # Stage 3: both mechanisms, each from the SAME starting weights.
    fusion_only = run_fusion_only_update(
        models, X_new, y_new, X_test_s, y_test, test_types, NOVEL_ATTACK_TYPE
    )
    print("\n" + fusion_only.summary())

    replay = run_incremental_update(
        models, buffer, X_new, y_new, X_test_s, y_test, test_types, NOVEL_ATTACK_TYPE
    )
    print("\n" + replay.summary())

    lines = [
        "# T3.5 — Incremental learning: absorbing a new attack type",
        "",
        "**Task:** T3.5 · **Generated by:** `scripts/evaluate_incremental_learning.py` · "
        "**Gate:** `TRD.md §9` \"Incremental learning doesn't regress old classes\"",
        "",
        f"> {POSITIVE_CLASS_STATEMENT}",
        "",
        "## Experiment",
        "",
        f"`{NOVEL_ATTACK_TYPE}` was held out of training entirely. A fresh ensemble was trained on",
        "the remaining attack types through the same leakage-free, session-split pipeline as T2.6,",
        f"then updated from **{len(X_new)} labelled windows** of the held-out type — deliberately a",
        "small sample, since the premise of incremental learning is that a full retrain's worth of",
        "data is not available.",
        "",
        f"The hold-out is `{NOVEL_ATTACK_TYPE}` because it is behaviourally unlike the Mirai and DoS",
        "flooding families that dominate IoTID20. An easy hold-out would make the experiment",
        "meaningless: the model would already detect it, and there would be nothing to learn.",
        "",
        "## Results",
        "",
        "| | Fusion priors only (rejected) | Replay + fine-tune (adopted) |",
        "|---|---:|---:|",
        f"| Trainable parameters | {fusion_only.trainable_parameters:,} | "
        f"{replay.trainable_parameters:,} |",
        f"| Recall on the NEW type, before | {fusion_only.novel_recall_before:.4f} | "
        f"{replay.novel_recall_before:.4f} |",
        f"| Recall on the NEW type, after | **{fusion_only.novel_recall_after:.4f}** | "
        f"**{replay.novel_recall_after:.4f}** |",
        f"| Mean P(Attack) on the NEW type | {fusion_only.novel_margin_before:.4f} → "
        f"{fusion_only.novel_margin_after:.4f} | {replay.novel_margin_before:.4f} → "
        f"**{replay.novel_margin_after:.4f}** |",
        f"| Recall on PRIOR types, before | {fusion_only.prior_recall_before:.4f} | "
        f"{replay.prior_recall_before:.4f} |",
        f"| Recall on PRIOR types, after | **{fusion_only.prior_recall_after:.4f}** | "
        f"**{replay.prior_recall_after:.4f}** |",
        f"| Δ prior recall (forgetting) | {fusion_only.prior_recall_delta:+.4f} | "
        f"{replay.prior_recall_delta:+.4f} |",
        f"| False-positive rate | {fusion_only.normal_fpr_before:.4f} → "
        f"{fusion_only.normal_fpr_after:.4f} | {replay.normal_fpr_before:.4f} → "
        f"{replay.normal_fpr_after:.4f} |",
        f"| Update time | {fusion_only.seconds:.1f}s | {replay.seconds:.1f}s |",
        f"| **`TRD.md §9` gate** | **{'PASS' if fusion_only.passes_gate() else 'FAIL'}** | "
        f"**{'PASS' if replay.passes_gate() else 'FAIL'}** |",
        "",
        "## ⚠️ Read the recall numbers with this caveat",
        "",
        f"**The model already detected `{NOVEL_ATTACK_TYPE}` at "
        f"{replay.novel_recall_before:.1%} recall before any update**, despite never having seen",
        "it. That is not a flaw in the mechanism — it is a property of the task: this is a",
        "**binary** Attack/Normal detector, and an unseen attack family can still look 'not normal'",
        "to a model trained on other attack families. The framing of an incremental update as",
        "*teaching the model to detect something it could not detect* does not hold here.",
        "",
        "Two consequences for how this experiment should be read:",
        "",
        f"1. **Recall is ceiling-limited.** Starting at {replay.novel_recall_before:.4f}, there is",
        "   at most 0.03 of headroom, so a small recall gain is not weak evidence of learning — it",
        "   is all the evidence the metric can carry. The **mean P(Attack)** row is included",
        "   precisely because it is not ceiling-limited and registers a strengthened belief that",
        "   recall cannot show.",
        f"2. **The false-positive rate is high ({replay.normal_fpr_before:.2f} before the update).**",
        "   A model biased toward predicting Attack will score well on any attack type, novel or",
        "   not. Part of that 97% pre-update recall is that bias rather than genuine",
        "   generalisation, and the FPR row must be read alongside every recall figure here.",
        "",
        "**What this experiment therefore does and does not establish.** It establishes that the",
        "replay mechanism absorbs new samples *without regressing prior classes* — which is exactly",
        f"what `TRD.md §9`'s gate asks. It does **not** establish that the mechanism can teach a",
        "genuinely undetectable attack type, because IoTID20 offers no such hold-out for a binary",
        "detector. Demonstrating that needs either a multi-class formulation or a dataset whose",
        "attack families are more behaviourally distinct — Edge-IIoTset (T3.6) is the first place",
        "to look.",
        "",
        "The gate requires two things together: prior-class recall must not fall by more than 0.01,",
        "**and** the update must actually have taught the model something — measured as a gain in",
        "either novel-type recall or novel-type mean probability. A mechanism that changed nothing",
        "would trivially preserve prior performance while being useless, so no-change is recorded as",
        "a failure rather than a pass.",
        "",
        "## Why the fusion-only mechanism was rejected",
        "",
        "`TRD.md §5.2` lists \"fine-tuning only the fusion layer\" first. It has exactly **three**",
        "adjustable parameters — one `BRANCH_PRIORS` scalar per branch — which can only re-mix",
        "opinions the branches already hold. When all three branches misclassify genuinely novel",
        "traffic, the correct answer is not among the inputs, and no re-weighting produces it.",
        "",
        "`docs/calibration_decision.md` §4 had already measured those same three priors on attack",
        "types the model *had* seen: +0.0017 validation F1, −0.0003 on test. A mechanism that cannot",
        "help on known types will not absorb an unknown one. The column above is the direct test.",
        "",
        "## Honest limits",
        "",
        "- **The held-out type is still IoTID20 traffic**, captured in the same conditions as the",
        "  training types, and was already detected at "
        f"{replay.novel_recall_before:.1%} before the update. See the caveat section above: this is",
        "  a much easier setting than a genuinely novel attack in deployment.",
        "- **A high base false-positive rate inflates every recall number here.** The ensemble",
        "  trained for this experiment sees fewer windows than T2.6's and is more Attack-biased; it",
        "  is not the model reported in `reports/phase2_results.md`.",
        f"- **One hold-out, one seed.** The experiment fixes `{NOVEL_ATTACK_TYPE}` and seed",
        f"  {RANDOM_STATE}. Rotating the held-out type would show whether the result is general;",
        "  that is a cheap extension and worth running before Review 3.",
        "- **Sessions are 100% label-pure** (`reports/phase2_results.md` §9), so a window of the new",
        "  type is never mixed with prior-type traffic. Real incremental learning would face that.",
        f"- **Fine-tuning touches {replay.trainable_parameters:,} parameters** across three branches.",
        "  That is far cheaper than a retrain, but it is not free, and it has not been measured on",
        "  the Raspberry Pi (T4.3). No claim is made here about on-device update cost.",
        "- **Replay requires retaining prior samples.** The buffer holds "
        f"{BUFFER_CAPACITY} windows of real traffic, which is a data-retention consideration in a",
        "  hospital setting even though these are flow statistics rather than payloads.",
        "",
        "## Parameters",
        "",
        f"`REPLAY_RATIO` {REPLAY_RATIO} · `BUFFER_CAPACITY` {BUFFER_CAPACITY} · "
        f"`FINE_TUNE_EPOCHS` {FINE_TUNE_EPOCHS} · `FINE_TUNE_LEARNING_RATE` "
        f"{FINE_TUNE_LEARNING_RATE} · seed {RANDOM_STATE}. All fixed in "
        "`src/adaptive/incremental.py` with their justifications.",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORTS_DIR / "t3_5_incremental_learning.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
