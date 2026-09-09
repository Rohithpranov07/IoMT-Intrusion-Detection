"""Guard against silent dependency drift.

Every version in `requirements.txt` is pinned exactly, and this test asserts the live environment
matches. That is unusual for a test suite, and it exists because the pins were NOT enough on their
own:

During Phase 3 an unpinned `pip install scipy` upgraded numpy 1.26 -> 2.4 and scikit-learn
1.3.2 -> 1.9.0 to satisfy a transitive dependency. `requirements.txt` already named the correct
versions; pip simply moved past them, because a pin in a file is not a constraint on a later
install. The drift went unnoticed until `imbalanced-learn` stopped importing several tasks later —
by which point results had been generated under a scikit-learn this project never intended to use.

`numpy` and `scikit-learn` are load-bearing for reproducibility: they determine the Random Forest
baseline, the feature ranking, the SMOTE output, and every split. A different version means
different numbers in `reports/`.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Packages whose version changes the NUMBERS this project reports, as opposed to tooling.
REPRODUCIBILITY_CRITICAL: frozenset[str] = frozenset(
    {"numpy", "scikit-learn", "imbalanced-learn", "scipy", "pandas", "tensorflow"}
)


def pinned_requirements() -> dict[str, str]:
    """Parse `requirements.txt` into a package -> exact-version mapping.

    Returns:
        Mapping of package name to pinned version.
    """
    pins: dict[str, str] = {}
    for raw in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#")[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip().lower()] = version.strip()
    return pins


def installed_versions() -> dict[str, str]:
    """Return the installed package versions of the running interpreter.

    Returns:
        Mapping of lowercased package name to version.
    """
    output = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
        capture_output=True, text=True, timeout=120,
    ).stdout
    return {
        line.split("==")[0].strip().lower(): line.split("==")[1].strip()
        for line in output.splitlines()
        if "==" in line
    }


def test_every_requirement_is_pinned_exactly() -> None:
    """An unpinned requirement is how the drift happened; none may remain."""
    for raw in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        assert "==" in line, f"requirement is not pinned to an exact version: {line!r}"
        assert not re.search(r"[<>~]", line), f"requirement uses a range: {line!r}"


@pytest.mark.parametrize("package", sorted(REPRODUCIBILITY_CRITICAL))
def test_reproducibility_critical_packages_match_their_pin(package: str) -> None:
    """These determine the numbers in `reports/`; a mismatch invalidates them.

    scikit-learn in particular decides the Random Forest baseline, the feature ranking, the SMOTE
    output and every split.
    """
    pins, installed = pinned_requirements(), installed_versions()
    assert package in pins, f"{package} is reproducibility-critical but not pinned"

    actual = installed.get(package)
    assert actual is not None, f"{package} is pinned but not installed"
    assert actual == pins[package], (
        f"{package} is {actual} but requirements.txt pins {pins[package]}. "
        "Results in reports/ were generated under the pinned version. Reinstall with "
        "`pip install -r requirements.txt` before trusting any regenerated number."
    )


def test_numpy_stays_below_2_for_the_sklearn_abi() -> None:
    """scikit-learn 1.3.2 is built against the numpy 1.x C ABI.

    Under numpy 2.x it raises "numpy.dtype size changed", which is how the drift finally surfaced.
    """
    import numpy

    assert numpy.__version__.startswith("1."), (
        f"numpy is {numpy.__version__}; scikit-learn 1.3.2 requires the 1.x ABI"
    )
