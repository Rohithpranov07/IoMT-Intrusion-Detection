"""Incremental learning: absorbing a new attack type without a full retrain (T3.5; TRD.md §5.2).

The defect this addresses
-------------------------
`PRD.md §2.1.4` (Objection #4): HIDS-IoMT can only take on a new attack type by retraining from
scratch, and `PRD.md` FR-9 requires this project to do better. `TRD.md §5.2` names two candidate
mechanisms and instructs us to pick one and document why.

MECHANISM CHOSEN: **replay-buffer fine-tuning of the branch classification heads**
-----------------------------------------------------------------------------------
`TRD.md §5.2`'s first candidate — "fine-tuning only the fusion layer (§3.3) on new samples while
freezing branch weights" — was **rejected on measured grounds**, not on preference:

  The fusion layer has exactly **three** adjustable parameters: `BRANCH_PRIORS`, one scalar per
  branch (`docs/architecture_decision.md` §3.1). Those weights can only re-mix opinions the
  branches already hold. If a novel attack is misclassified by all three branches — which is the
  normal case for genuinely novel traffic — then no re-weighting of their outputs can produce a
  correct answer, because the correct answer is not among the inputs. Three degrees of freedom
  cannot represent a new decision boundary.

  This is not merely a theoretical objection. `docs/calibration_decision.md` §4 measured a fit of
  those same three priors on the validation fold: it improved validation F1 by 0.0017 and **cost**
  0.0003 on test, and drove one prior to zero. If fitting them cannot help on attack types the
  model has already seen, it will not absorb one it has not.

  `run_fusion_only_update()` implements that rejected mechanism anyway, so the comparison in
  `reports/t3_5_incremental_learning.md` is measured rather than asserted.

The adopted mechanism is `TRD.md §5.2`'s second candidate, made concrete:

  1. **Freeze the feature extractors.** Only layers whose names end in `TRAINABLE_LAYER_SUFFIXES`
     stay trainable — the classification head and the embedding projection. The convolutional,
     recurrent, and attention stacks are what encode "what network traffic looks like", they were
     learned from far more data than any incremental update will have, and they are the expensive
     part to update on a Raspberry Pi (T4.2).
  2. **Mix new samples with a replay buffer** of prior-class examples, so the update sees the old
     classes it must not forget. This is the direct guard against catastrophic forgetting, which is
     what `TRD.md §9`'s "Incremental learning doesn't regress old classes" gate tests.
  3. **Fine-tune briefly at a low learning rate**, to bound how far weights can move.

Exact parameters
----------------
    BUFFER_CAPACITY          = 2000   windows retained from prior classes. Small enough to sit on
                                      a fog node; large enough that replay is not itself an
                                      imbalanced sample. Kept CLASS-BALANCED by reservoir sampling
                                      per class, so a rare prior attack type is not crowded out by
                                      a common one — the rare types are exactly the ones forgetting
                                      would erase first.
    REPLAY_RATIO             = 2.0    replay windows drawn per new window. Below 1.0 the update is
                                      dominated by the new class and forgetting sets in; far above
                                      2.0 the new class is drowned out and never learned. 2.0 is
                                      this project's choice and the value the ablation in
                                      `reports/t3_5_incremental_learning.md` varies.
    FINE_TUNE_EPOCHS         = 8
    FINE_TUNE_LEARNING_RATE  = 1e-4   ten times below training's 1e-3 (`train_utils`), to bound
                                      drift from the frozen representation.
    FINE_TUNE_BATCH_SIZE     = 64     smaller than training's 256: incremental updates are small,
                                      and more gradient steps per epoch helps at a low rate.
    TRAINABLE_LAYER_SUFFIXES = ("_head", "_embedding")
    RANDOM_STATE             = 42

Positive class: index 1 = **Attack** (`TRD.md §2.3`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from tensorflow import keras

from src.config import RANDOM_STATE
from src.models.fusion import confidence_weighted_fusion

logger = logging.getLogger(__name__)

BUFFER_CAPACITY: int = 2000
REPLAY_RATIO: float = 2.0
FINE_TUNE_EPOCHS: int = 8
FINE_TUNE_LEARNING_RATE: float = 1e-4
FINE_TUNE_BATCH_SIZE: int = 64
TRAINABLE_LAYER_SUFFIXES: tuple[str, ...] = ("_head", "_embedding")


class ReplayBuffer:
    """A small, class-balanced store of prior-class windows.

    Balance is maintained **per class**, not globally. A globally-sampled buffer would fill with
    whatever attack type happened to be most common, and the rare prior types — the ones an update
    would erase first — would be the ones missing from the replay that is supposed to protect them.

    Attributes:
        capacity: total windows retained across all classes.
        classes_: the class keys currently stored.
    """

    def __init__(self, capacity: int = BUFFER_CAPACITY, random_state: int = RANDOM_STATE) -> None:
        self.capacity = capacity
        self._rng = np.random.default_rng(random_state)
        self._store: dict[object, list[np.ndarray]] = {}
        self._labels: dict[object, list[int]] = {}

    @property
    def classes_(self) -> list[object]:
        """Return the class keys currently held."""
        return sorted(self._store, key=str)

    def __len__(self) -> int:
        """Return the total number of windows stored."""
        return sum(len(v) for v in self._store.values())

    def add(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_keys: np.ndarray | list[object],
    ) -> "ReplayBuffer":
        """Add windows, keeping the buffer class-balanced within `capacity`.

        Args:
            X: windows, `(n, sequence_length, n_features)`.
            y: binary labels (1 = Attack).
            class_keys: the class each window belongs to — the attack sub-type, or the string
                "Normal". Balance is maintained across these keys.

        Returns:
            self.

        Raises:
            ValueError: if the inputs disagree in length.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).ravel()
        keys = np.asarray(class_keys, dtype=object).ravel()
        if not (len(X) == len(y) == len(keys)):
            raise ValueError(f"lengths differ: X={len(X)}, y={len(y)}, class_keys={len(keys)}")

        for key in np.unique(keys):
            mask = keys == key
            self._store.setdefault(key, []).extend(list(X[mask]))
            self._labels.setdefault(key, []).extend(list(y[mask].astype(int)))

        self._rebalance()
        logger.info(
            "Replay buffer holds %d windows across %d classes", len(self), len(self._store)
        )
        return self

    def _rebalance(self) -> None:
        """Trim each class to an equal share of `capacity`, sampling without replacement."""
        if not self._store:
            return
        per_class = max(1, self.capacity // len(self._store))
        for key in self._store:
            windows, labels = self._store[key], self._labels[key]
            if len(windows) > per_class:
                keep = self._rng.choice(len(windows), size=per_class, replace=False)
                self._store[key] = [windows[i] for i in keep]
                self._labels[key] = [labels[i] for i in keep]

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Draw `n` windows, spread as evenly as possible across the stored classes.

        Args:
            n: number of windows requested. Returns fewer if the buffer holds fewer.

        Returns:
            Tuple of `(X, y)`.
        """
        if not self._store:
            return np.empty((0,)), np.empty((0,))

        per_class = max(1, n // len(self._store))
        windows: list[np.ndarray] = []
        labels: list[int] = []
        for key in self.classes_:
            available = len(self._store[key])
            take = min(per_class, available)
            picked = self._rng.choice(available, size=take, replace=False)
            windows.extend(self._store[key][i] for i in picked)
            labels.extend(self._labels[key][i] for i in picked)

        return np.stack(windows).astype(np.float32), np.asarray(labels, dtype=np.int8)


@dataclass
class IncrementalResult:
    """Before/after evidence for one incremental update.

    `TRD.md §9`'s gate is that prior-class detection must not regress, so the prior-class numbers
    are first-class fields rather than something a caller has to reconstruct.

    Attributes:
        mechanism: "replay_finetune" or "fusion_only".
        new_attack_type: the attack type introduced.
        prior_recall_before / prior_recall_after: recall on PRIOR attack types.
        prior_recall_delta: after minus before. Negative is forgetting.
        novel_recall_before / novel_recall_after: recall on the NEW attack type.
        novel_margin_before / novel_margin_after: mean fused P(Attack) on the NEW type's windows.
            Recall saturates — a binary detector trained on other attack families often already
            flags a held-out one, leaving recall pinned near 1.0 with no room to show learning.
            The mean probability is not ceiling-limited, so it registers whether the update
            actually strengthened the model's belief. Report BOTH: a recall that cannot move is
            not evidence that nothing was learned.
        normal_fpr_before / normal_fpr_after: false-positive rate on Normal traffic, so a recall
            gain bought purely by flagging everything is visible.
        n_new_samples: labelled windows of the new type used for the update.
        n_replay_samples: replay windows mixed in.
        trainable_parameters: parameters the update was allowed to move.
        seconds: wall-clock time of the update.
    """

    mechanism: str
    new_attack_type: str
    prior_recall_before: float
    prior_recall_after: float
    novel_recall_before: float
    novel_recall_after: float
    normal_fpr_before: float
    normal_fpr_after: float
    n_new_samples: int
    n_replay_samples: int
    trainable_parameters: int
    seconds: float
    novel_margin_before: float = float("nan")
    novel_margin_after: float = float("nan")
    regression_tolerance: float = 0.01

    @property
    def prior_recall_delta(self) -> float:
        """Return the change in prior-class recall. Negative means forgetting."""
        return self.prior_recall_after - self.prior_recall_before

    @property
    def novel_recall_delta(self) -> float:
        """Return the change in recall on the newly-introduced attack type."""
        return self.novel_recall_after - self.novel_recall_before

    @property
    def novel_margin_delta(self) -> float:
        """Return the change in mean P(Attack) on the new type's windows."""
        return self.novel_margin_after - self.novel_margin_before

    def passes_gate(self) -> bool:
        """Return whether `TRD.md §9`'s no-regression gate is satisfied.

        Requires prior-class recall not to fall by more than `regression_tolerance`, AND the update
        to have actually taught the model something. A mechanism that changed nothing at all would
        trivially preserve prior performance while being useless.
        """
        margin_improved = (
            self.novel_margin_delta > 0.0 if np.isfinite(self.novel_margin_delta) else False
        )
        learned = self.novel_recall_delta > 0.0 or margin_improved
        return self.prior_recall_delta >= -self.regression_tolerance and learned

    def summary(self) -> str:
        """Render a printable before/after block."""
        return (
            f"Incremental update — {self.mechanism} (new type: {self.new_attack_type})\n"
            f"  trainable parameters      : {self.trainable_parameters:,}\n"
            f"  new samples / replay      : {self.n_new_samples:,} / {self.n_replay_samples:,}\n"
            f"  recall on NEW type        : {self.novel_recall_before:.4f} -> "
            f"{self.novel_recall_after:.4f}  ({self.novel_recall_delta:+.4f})\n"
            f"  mean P(Attack) on NEW type: {self.novel_margin_before:.4f} -> "
            f"{self.novel_margin_after:.4f}  ({self.novel_margin_delta:+.4f})\n"
            f"  recall on PRIOR types     : {self.prior_recall_before:.4f} -> "
            f"{self.prior_recall_after:.4f}  ({self.prior_recall_delta:+.4f})\n"
            f"  false-positive rate       : {self.normal_fpr_before:.4f} -> "
            f"{self.normal_fpr_after:.4f}\n"
            f"  update time               : {self.seconds:.1f}s\n"
            f"  TRD §9 no-regression gate : {'PASS' if self.passes_gate() else 'FAIL'}"
        )


def set_trainable_layers(
    model: keras.Model, suffixes: tuple[str, ...] = TRAINABLE_LAYER_SUFFIXES
) -> int:
    """Freeze every layer except those whose names end in one of `suffixes`.

    Args:
        model: the branch to configure, modified in place.
        suffixes: layer-name suffixes that stay trainable.

    Returns:
        The number of trainable parameters remaining.
    """
    for layer in model.layers:
        layer.trainable = layer.name.endswith(suffixes)
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


def fuse_predict(models: dict[str, keras.Model], X: np.ndarray) -> np.ndarray:
    """Return fused predictions for `X` using the deployed fusion rule.

    Args:
        models: branch name to model.
        X: windows, `(n, sequence_length, n_features)`.

    Returns:
        `(n,)` predicted labels; 1 = Attack.
    """
    names = list(models)
    return confidence_weighted_fusion(
        [models[name].predict(X, verbose=0) for name in names], names
    ).predictions


def _rates(
    models: dict[str, keras.Model],
    X: np.ndarray,
    y: np.ndarray,
    types: np.ndarray,
    novel_type: str,
) -> tuple[float, float, float, float]:
    """Compute prior-type recall, novel-type recall, the Normal FPR, and the novel-type margin.

    Args:
        models: branch name to model.
        X: evaluation windows.
        y: binary labels.
        types: per-window attack sub-type (or "Normal").
        novel_type: the sub-type treated as newly introduced.

    Returns:
        Tuple of `(prior_recall, novel_recall, normal_fpr, novel_mean_attack_probability)`.
    """
    names = list(models)
    fused = confidence_weighted_fusion(
        [models[name].predict(X, verbose=0) for name in names], names
    )
    predictions = fused.predictions
    attack_probability = fused.probabilities[:, 1]
    y = np.asarray(y).ravel()
    types = np.asarray(types, dtype=object).ravel()

    is_attack = y == 1
    prior = is_attack & (types != novel_type)
    novel = is_attack & (types == novel_type)
    normal = ~is_attack

    return (
        float(predictions[prior].mean()) if prior.any() else float("nan"),
        float(predictions[novel].mean()) if novel.any() else float("nan"),
        float(predictions[normal].mean()) if normal.any() else float("nan"),
        float(attack_probability[novel].mean()) if novel.any() else float("nan"),
    )


def run_incremental_update(
    models: dict[str, keras.Model],
    buffer: ReplayBuffer,
    X_new: np.ndarray,
    y_new: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    eval_types: np.ndarray,
    novel_type: str,
    replay_ratio: float = REPLAY_RATIO,
    epochs: int = FINE_TUNE_EPOCHS,
    learning_rate: float = FINE_TUNE_LEARNING_RATE,
) -> IncrementalResult:
    """Absorb a new attack type by replay-buffer fine-tuning of the branch heads.

    Args:
        models: trained branches, updated in place.
        buffer: replay buffer holding prior-class windows.
        X_new: windows of the newly-observed attack type.
        y_new: their labels (all 1 = Attack).
        X_eval: held-out evaluation windows spanning prior types, the new type, and Normal.
        y_eval: their binary labels.
        eval_types: per-window sub-type for `X_eval`.
        novel_type: the sub-type being introduced.
        replay_ratio: replay windows drawn per new window.
        epochs: fine-tuning epochs.
        learning_rate: fine-tuning learning rate.

    Returns:
        An `IncrementalResult` with before/after measurements.
    """
    import time

    started = time.perf_counter()
    prior_before, novel_before, fpr_before, margin_before = _rates(
        models, X_eval, y_eval, eval_types, novel_type
    )

    X_replay, y_replay = buffer.sample(int(replay_ratio * len(X_new)))
    if len(X_replay):
        X_update = np.concatenate([np.asarray(X_new, dtype=np.float32), X_replay])
        y_update = np.concatenate([np.asarray(y_new).ravel(), y_replay])
    else:
        X_update, y_update = np.asarray(X_new, dtype=np.float32), np.asarray(y_new).ravel()

    trainable_total = 0
    for name, model in models.items():
        trainable_total += set_trainable_layers(model)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            X_update,
            keras.utils.to_categorical(y_update, num_classes=2),
            epochs=epochs,
            batch_size=FINE_TUNE_BATCH_SIZE,
            verbose=0,
        )
        logger.info("Fine-tuned %s on %d windows", name, len(X_update))

    prior_after, novel_after, fpr_after, margin_after = _rates(
        models, X_eval, y_eval, eval_types, novel_type
    )

    result = IncrementalResult(
        mechanism="replay_finetune",
        new_attack_type=novel_type,
        prior_recall_before=prior_before,
        prior_recall_after=prior_after,
        novel_recall_before=novel_before,
        novel_recall_after=novel_after,
        novel_margin_before=margin_before,
        novel_margin_after=margin_after,
        normal_fpr_before=fpr_before,
        normal_fpr_after=fpr_after,
        n_new_samples=len(X_new),
        n_replay_samples=len(X_replay),
        trainable_parameters=trainable_total,
        seconds=time.perf_counter() - started,
    )
    logger.info("%s", result.summary())
    return result


def run_fusion_only_update(
    models: dict[str, keras.Model],
    X_new: np.ndarray,
    y_new: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    eval_types: np.ndarray,
    novel_type: str,
    grid_steps: int = 11,
) -> IncrementalResult:
    """`TRD.md §5.2`'s REJECTED first candidate, implemented so the comparison is measured.

    Fits only the fusion layer's three `BRANCH_PRIORS` on the new samples, leaving every branch
    weight frozen. Included to demonstrate the argument in this module's docstring rather than
    merely assert it: three scalars can re-mix existing branch opinions but cannot represent a new
    decision boundary.

    Args:
        models: trained branches. **Not modified** — only the fusion priors are fitted.
        X_new: windows of the newly-observed attack type.
        y_new: their labels.
        X_eval: held-out evaluation windows.
        y_eval: their binary labels.
        eval_types: per-window sub-type for `X_eval`.
        novel_type: the sub-type being introduced.
        grid_steps: resolution of the per-prior grid search.

    Returns:
        An `IncrementalResult` with `mechanism="fusion_only"`.
    """
    import itertools
    import time

    started = time.perf_counter()
    names = list(models)
    prior_before, novel_before, fpr_before, margin_before = _rates(
        models, X_eval, y_eval, eval_types, novel_type
    )

    new_probabilities = [models[n].predict(X_new, verbose=0) for n in names]
    y_new_array = np.asarray(y_new).ravel()

    grid = np.linspace(0.0, 1.0, grid_steps)
    best_priors, best_accuracy = [1.0] * len(names), -1.0
    for candidate in itertools.product(grid, repeat=len(names)):
        if sum(candidate) == 0:
            continue
        fused = confidence_weighted_fusion(
            new_probabilities, names, branch_priors=list(candidate)
        )
        accuracy = float((fused.predictions == y_new_array).mean())
        if accuracy > best_accuracy:
            best_accuracy, best_priors = accuracy, list(candidate)

    eval_probabilities = [models[n].predict(X_eval, verbose=0) for n in names]
    fused_after = confidence_weighted_fusion(
        eval_probabilities, names, branch_priors=best_priors
    )
    predictions = fused_after.predictions

    y_eval_array = np.asarray(y_eval).ravel()
    types = np.asarray(eval_types, dtype=object).ravel()
    is_attack = y_eval_array == 1
    prior_mask = is_attack & (types != novel_type)
    novel_mask = is_attack & (types == novel_type)
    normal_mask = ~is_attack

    logger.info("Best fusion priors on the new type: %s", dict(zip(names, best_priors)))
    result = IncrementalResult(
        mechanism="fusion_only",
        new_attack_type=novel_type,
        prior_recall_before=prior_before,
        prior_recall_after=float(predictions[prior_mask].mean()),
        novel_recall_before=novel_before,
        novel_recall_after=float(predictions[novel_mask].mean()),
        novel_margin_before=margin_before,
        novel_margin_after=float(fused_after.probabilities[novel_mask, 1].mean()),
        normal_fpr_before=fpr_before,
        normal_fpr_after=float(predictions[normal_mask].mean()),
        n_new_samples=len(X_new),
        n_replay_samples=0,
        trainable_parameters=len(names),  # three scalars, and that is the whole point
        seconds=time.perf_counter() - started,
    )
    logger.info("%s", result.summary())
    return result
