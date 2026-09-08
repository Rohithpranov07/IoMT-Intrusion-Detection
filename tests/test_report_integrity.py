"""Repository-wide integrity checks on the reports (Build-Instructions §A.1, T4.4 VERIFY).

`Build-Instructions.md` §A.1 states a rule that no unit test would normally touch:

    "Never compare this project's results to the base paper's raw numbers without the
     metric-inversion caveat. ... omitting that caveat when convenient (e.g., because this
     project's own numbers are lower) is an integrity failure, not a formatting nicety."

A rule that depends on remembering gets forgotten exactly when the numbers are inconvenient, so it
is enforced here instead. These tests scan every Markdown file this project generates and fail the
build if a base-paper figure appears without its caveat, or if a metrics table appears without
stating its positive class.

They are deliberately about *documents*, not code. That is unusual for a test suite, and it is the
point: the documents are the deliverable.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The base paper's headline figures (`PRD.md §2.1.3`). Any of these appearing in a generated
#: document must be accompanied by the inversion caveat.
BASE_PAPER_FIGURES: tuple[str, ...] = ("99.92", "99.91", "99.99", "99.95")

#: Terms that, together, constitute the caveat. At least one from each group must appear near a cited figure.
INVERSION_TERMS: tuple[str, ...] = ("swap", "invert", "inversion")
CONTAMINATION_TERMS: tuple[str, ...] = ("contaminat", "leak", "synthetic")

#: Ways this repository refers to the base paper. A paragraph containing one of
#: `BASE_PAPER_FIGURES` is only a CITATION if it also names the paper -- otherwise the check fires
#: on coincidences. It did: `gnn_go_nogo.md` reports "99.92% of nodes have degree <= 2", which has
#: nothing to do with HIDS-IoMT, and an over-broad check would have forced a nonsensical caveat
#: onto a graph statistic. A rule that cries wolf gets switched off.
BASE_PAPER_NAMES: tuple[str, ...] = ("hids-iomt", "base paper", "berguiga")

#: Documents this project generates. `PRD.md`, `TRD.md` and `Build-Instructions.md` are the team's
#: source specifications and are excluded -- they are inputs, not outputs, and PRD.md §10 is itself
#: where the caveat is derived.
SOURCE_SPECS: frozenset[str] = frozenset(
    {"PRD.md", "TRD.md", "Build-Instructions.md", "OPERATING_CONTRACT.md"}
)


def generated_markdown() -> list[Path]:
    """Return every Markdown document this project generates.

    Returns:
        Paths under `reports/` and `docs/`, excluding the team's source specifications.
    """
    paths: list[Path] = []
    for directory in ("reports", "docs"):
        paths += sorted((REPO_ROOT / directory).glob("*.md"))
    return [p for p in paths if p.name not in SOURCE_SPECS]


def paragraphs_containing(text: str, needle: str) -> list[str]:
    """Return the blank-line-separated paragraphs of `text` that contain `needle`.

    Args:
        text: the document.
        needle: the substring to locate.

    Returns:
        Matching paragraphs.
    """
    return [block for block in re.split(r"\n\s*\n", text) if needle in block]


# --- The rule §A.1 exists to enforce ------------------------------------------------------------


@pytest.mark.parametrize("document", generated_markdown(), ids=lambda p: p.name)
def test_base_paper_figures_always_carry_the_caveat(document: Path) -> None:
    """T4.4 VERIFY: no base-paper number may appear without the metric-inversion caveat.

    Checked per PARAGRAPH, not per file: `Build-Instructions.md` §A.1 requires the caveat "in the
    same paragraph", because a caveat buried at the foot of a long document does not travel when
    someone copies a table into a slide.
    """
    text = document.read_text(encoding="utf-8")

    for block in re.split(r"\n\s*\n", text):
        lowered = block.lower()
        figures_present = [f for f in BASE_PAPER_FIGURES if f in block]
        if not figures_present:
            continue

        # A figure is a CITATION when the paragraph names the base paper, or when it carries three
        # or more of the four headline figures together (unmistakably the published set).
        names_paper = any(name in lowered for name in BASE_PAPER_NAMES)
        if not names_paper and len(figures_present) < 3:
            continue

        # A paragraph stating the prohibition ("never cite ... without ...") is compliant: it is
        # the rule, not a breach of it.
        if any(word in lowered for word in ("never", "must not", "forbid", "without")):
            continue

        assert any(term in lowered for term in INVERSION_TERMS), (
            f"{document.name}: a paragraph cites the base paper's "
            f"{', '.join(figures_present)} without the metric-inversion caveat.\n\n{block[:400]}"
        )


def test_final_comparison_states_both_defects_where_it_cites_the_paper() -> None:
    """The caveat has two halves — swapped labels AND a contaminated fold. Both must appear."""
    text = (REPO_ROOT / "reports" / "final_comparison.md").read_text(encoding="utf-8").lower()

    citing = [b for b in re.split(r"\n\s*\n", text) if "99.92" in b]
    assert citing, "final_comparison.md does not cite the base paper at all"

    for block in citing:
        if any(word in block for word in ("never", "must not", "forbid")):
            continue
        assert any(term in block for term in INVERSION_TERMS), "missing the inversion caveat"
        assert any(term in block for term in CONTAMINATION_TERMS), (
            "missing the contaminated-evaluation-fold half of the caveat"
        )


# --- The positive-class convention (TRD.md §2.3) ------------------------------------------------


@pytest.mark.parametrize("document", generated_markdown(), ids=lambda p: p.name)
def test_documents_reporting_metrics_state_their_positive_class(document: Path) -> None:
    """Objection #3 is a labelling failure; a table without its convention is ambiguous."""
    text = document.read_text(encoding="utf-8")

    reports_metrics = bool(
        re.search(r"\|\s*(Precision|Recall|F1)\s*\|", text, flags=re.IGNORECASE)
    )
    if not reports_metrics:
        pytest.skip("no metrics table in this document")

    lowered = text.lower()
    assert "positive class" in lowered, (
        f"{document.name} reports precision/recall/F1 without stating its positive class "
        "(TRD.md §2.3, Objection #3)"
    )
    assert "attack" in lowered, f"{document.name} does not name Attack as the positive class"


# --- The GNN gate (Build-Instructions §A.1 rule 4) -----------------------------------------------


def test_no_gnn_branch_reappeared() -> None:
    """T2.7 returned no-go; the branch must be absent from the codebase, not merely unused."""
    assert not (REPO_ROOT / "src" / "models" / "gnn_branch.py").exists()
    assert (REPO_ROOT / "reports" / "gnn_go_nogo.md").exists()

    decision = (REPO_ROOT / "reports" / "gnn_go_nogo.md").read_text(encoding="utf-8")
    assert "NO-GO" in decision, "the go/no-go report must state a decision explicitly"


# --- No unmeasured deployment claim (§A.1 rule 3) ------------------------------------------------


@pytest.mark.parametrize("document", generated_markdown(), ids=lambda p: p.name)
def test_no_document_claims_pi_performance_without_a_pi(document: Path) -> None:
    """§A.1 rule 3: no latency or throughput claim without a real Pi measurement.

    Until `reports/deployment_benchmark.md` exists — written only by a run on real hardware — no
    document may assert a Pi latency figure. `DRAFT_` files are exempt: they exist precisely to be
    labelled as not-evidence.
    """
    if document.name.startswith("DRAFT_"):
        pytest.skip("drafts are explicitly stamped as not deployment evidence")

    real_benchmark_exists = (REPO_ROOT / "reports" / "deployment_benchmark.md").exists()
    if real_benchmark_exists:
        pytest.skip("a real Pi benchmark exists, so Pi timings may be cited")

    text = document.read_text(encoding="utf-8")
    # A figure in milliseconds attributed to the Pi, in a sentence that is not a disclaimer.
    for block in re.split(r"\n\s*\n", text):
        lowered = block.lower()
        if "raspberry pi" not in lowered:
            continue
        if not re.search(r"\d+(\.\d+)?\s*(ms|milliseconds)\b", lowered):
            continue
        assert any(
            word in lowered
            for word in ("no ", "not ", "refus", "until", "would", "must", "cannot", "500")
        ), (
            f"{document.name} appears to state a Raspberry Pi timing, but no measurement from "
            f"real hardware exists.\n\n{block[:400]}"
        )
