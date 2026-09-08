"""Tests for the plain-language feature glossary (Build-Instructions T3.3).

T3.3's VERIFY block is a human read that no test can perform. What these tests DO establish is the
defect that gate would have surfaced first: an explanation naming `Init_Bwd_Win_Byts` is traceable
to the data but not comprehensible to the clinician `PRD.md §4` says is the audience.

`test_every_selected_feature_has_a_description` is the substantive one — it fails the build if a
feature can be cited in an alert without a plain-language rendering.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from src.config import ARTIFACTS_DIR, NEGATIVE_CLASS_NAME, POSITIVE_CLASS_NAME
from src.xai.feature_glossary import (
    FEATURE_DESCRIPTIONS,
    describe_feature,
    glossary_coverage,
    render_feature,
)
from src.xai.shap_explainer import Explanation, FeatureAttribution

REPO_ROOT = Path(__file__).resolve().parents[1]


def selected_features() -> list[str]:
    """Return the features the deployed model actually uses, if exported."""
    path = ARTIFACTS_DIR / "feature_names.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["feature_names"]


# --- Coverage -----------------------------------------------------------------------------------


@pytest.mark.skipif(not selected_features(), reason="run notebook 03 to export feature names")
def test_every_selected_feature_has_a_description() -> None:
    """Any feature the model can cite must be renderable in plain language.

    A gap here means an alert can name a column a clinician cannot interpret — the readability
    defect T3.3's reader test exists to catch.
    """
    covered, missing = glossary_coverage(selected_features())
    assert not missing, f"{len(missing)} selected features have no description: {missing}"
    assert covered == 1.0


def test_coverage_reports_gaps_rather_than_inventing_text() -> None:
    """An unknown feature must return None, not a plausible-sounding guess.

    A fabricated description of a network measurement is worse than an honest gap: the reader
    cannot tell the difference, and would act on it.
    """
    assert describe_feature("Totally_Made_Up_Column") is None
    covered, missing = glossary_coverage(["Dst_Port", "Totally_Made_Up_Column"])
    assert missing == ["Totally_Made_Up_Column"]
    assert covered == pytest.approx(0.5)


def test_render_keeps_the_column_name_for_traceability() -> None:
    """T3.1's VERIFY block requires a reader map a feature back to the data.

    The description is ADDED for readability, never SUBSTITUTED for provenance.
    """
    rendered = render_feature("Init_Bwd_Win_Byts")
    assert "Init_Bwd_Win_Byts" in rendered
    assert "ready to receive" in rendered


def test_undescribed_feature_renders_as_its_bare_name() -> None:
    """Falling back to the column name keeps the alert usable when the glossary is incomplete."""
    assert render_feature("Unknown_Column") == "Unknown_Column"


# --- The descriptions themselves ------------------------------------------------------------------


def test_descriptions_avoid_capture_tool_jargon() -> None:
    """"Forward"/"backward" is CICFlowMeter's convention, not a hospital IT lead's vocabulary.

    Matched on WORD BOUNDARIES, not as substrings. A plain substring search flagged
    "immediately" for containing "iat" — a check that fires on ordinary English would either be
    silenced or would push the descriptions toward stilted wording to appease it.
    """
    jargon = ("fwd", "bwd", "iat", "subflow", "flag_cnt", "seg_size", "tot_", "pkt")
    pattern = re.compile(r"\b(" + "|".join(jargon) + r")\b")

    for feature, description in FEATURE_DESCRIPTIONS.items():
        found = pattern.findall(description.lower())
        assert not found, (
            f"{feature}'s description leaks the raw naming scheme {found}: {description!r}"
        )


def test_descriptions_are_sentences_not_restated_column_names() -> None:
    """A description that just re-spells the identifier adds nothing."""
    for feature, description in FEATURE_DESCRIPTIONS.items():
        assert len(description.split()) >= 4, f"{feature}'s description is too terse"
        assert "_" not in description.replace(feature, ""), (
            f"{feature}'s description contains an underscore-style identifier"
        )


def test_direction_is_expressed_relative_to_the_device() -> None:
    """The reader is protecting a device; they should not have to learn capture-direction terms."""
    directional = [
        d for f, d in FEATURE_DESCRIPTIONS.items() if f.startswith(("Fwd_", "Bwd_", "Init_Bwd"))
    ]
    assert directional
    assert all("device" in d.lower() for d in directional)


# --- The rendered alert ----------------------------------------------------------------------------


def _explanation(attributions: list[FeatureAttribution], predicted: int = 1) -> Explanation:
    """Build an Explanation carrying the given attributions."""
    return Explanation(
        predicted_label=predicted,
        confidence=0.97,
        top_features=attributions,
        branch_weights=np.array([0.4, 0.3, 0.3]),
        branch_names=["cnn", "bilstm", "transformer"],
        timestep_attributions=np.zeros((10, 2)),
        feature_names=["Dst_Port", "SYN_Flag_Cnt"],
    )


def test_summary_splits_supporting_from_opposing_evidence() -> None:
    """Resolves the open question from `reports/t3_1_shap_examples.md`.

    Ranking by absolute influence is correct and makes the shares sum to 100%, but it let the
    top-ranked feature argue against its own verdict, which reads as contradictory. Splitting keeps
    every feature and its true direction while removing the contradiction.
    """
    summary = _explanation([
        FeatureAttribution(1, "Dst_Port", +0.5, 0.5, 3),
        FeatureAttribution(2, "SYN_Flag_Cnt", -0.4, 0.4, 5),
    ]).to_summary()

    assert "What pointed to Attack:" in summary
    assert "What argued against it" in summary
    # Nothing is dropped to make the alert tidier -- both features must still appear.
    assert "Dst_Port" in summary and "SYN_Flag_Cnt" in summary


def test_summary_omits_an_empty_section() -> None:
    """When every feature agrees, no 'argued against' heading should appear."""
    summary = _explanation([
        FeatureAttribution(1, "Dst_Port", +0.5, 0.5, 3),
        FeatureAttribution(2, "SYN_Flag_Cnt", +0.4, 0.4, 5),
    ]).to_summary()

    assert "What pointed to Attack:" in summary
    assert "What argued against it" not in summary


def test_summary_renders_features_in_plain_language() -> None:
    """The alert must not present a bare column name as its explanation."""
    summary = _explanation([FeatureAttribution(1, "Init_Bwd_Win_Byts", 0.5, 1.0, 0)]).to_summary()
    assert "ready to receive" in summary
    assert "Init_Bwd_Win_Byts" in summary  # provenance retained


def test_direction_split_follows_the_verdict_not_the_sign() -> None:
    """For a Normal verdict, a NEGATIVE attribution is the supporting evidence."""
    summary = _explanation(
        [FeatureAttribution(1, "Dst_Port", -0.5, 1.0, 0)], predicted=0
    ).to_summary()

    assert f"What pointed to {NEGATIVE_CLASS_NAME.title()}:" in summary
    assert "What argued against it" not in summary


@pytest.mark.skipif(
    not (REPO_ROOT / "reports" / "t3_1_shap_examples.md").exists(),
    reason="run scripts/demo_shap_explanation.py first",
)
def test_generated_examples_contain_no_bare_column_names() -> None:
    """Every feature cited in a real alert must arrive with its plain-language rendering."""
    text = (REPO_ROOT / "reports" / "t3_1_shap_examples.md").read_text(encoding="utf-8")

    # Inside the rendered alert blocks, a column name must always be preceded by a description.
    for match in re.finditer(r"^  \d+\. (.+)$", text, flags=re.MULTILINE):
        line = match.group(1)
        if "(" not in line:
            pytest.fail(f"alert cites a bare column name with no description: {line!r}")
        description = line.split("(")[0].strip()
        assert len(description.split()) >= 4, f"description too terse in alert line: {line!r}"


def test_positive_class_is_named_in_every_alert() -> None:
    """`TRD.md §2.3` -- an alert without its convention is ambiguous."""
    summary = _explanation([FeatureAttribution(1, "Dst_Port", 0.5, 1.0, 0)]).to_summary()
    assert POSITIVE_CLASS_NAME in summary
