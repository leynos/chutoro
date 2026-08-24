"""Contract between the fast-tier Makefile target and the fast-tier docs.

``make kani`` is the pull-request gate, so its meaning depends on which
harnesses the ``kani:`` target actually runs. The MST harness module
documents that both of its proofs are fast-tier; this test derives the
harness names from that source file and asserts the Makefile runs each,
so the tier decision cannot drift from the documented one. Deriving the
names rather than restating them keeps the contract self-maintaining: a
proof added to the module fails here until the Makefile carries it.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE_PATH = REPO_ROOT / "Makefile"

#: Harness modules whose module docs declare every proof fast-tier.
FAST_TIER_SOURCES = (REPO_ROOT / "chutoro-core" / "src" / "mst" / "kani_harness.rs",)

PROOF_RE = re.compile(r"^\s*fn\s+(verify_\w+)", re.MULTILINE)


def _fast_tier_harnesses() -> list[str]:
    """Return every proof name declared in the fast-tier harness modules."""
    names: list[str] = []
    for source in FAST_TIER_SOURCES:
        names.extend(PROOF_RE.findall(source.read_text(encoding="utf-8")))
    return names


@pytest.fixture(scope="module")
def kani_target() -> str:
    """Return the recipe body of the Makefile's ``kani:`` target."""
    lines = MAKEFILE_PATH.read_text(encoding="utf-8").splitlines()
    body: list[str] = []
    in_target = False
    for line in lines:
        if re.match(r"^kani:", line):
            in_target = True
            continue
        if in_target:
            if line.startswith("\t"):
                body.append(line)
            else:
                break
    assert body, "the Makefile must define a kani: target with a recipe"
    return "\n".join(body)


def test_fast_tier_sources_declare_harnesses() -> None:
    """Guard the derivation: an empty scan would pass everything vacuously."""
    harnesses = _fast_tier_harnesses()
    assert harnesses, (
        f"no proofs found in {[str(p) for p in FAST_TIER_SOURCES]}; "
        "the derivation regex or source list is broken"
    )


def test_make_kani_runs_every_fast_tier_harness(kani_target: str) -> None:
    """Each documented fast-tier proof appears in the gating target."""
    missing = [
        name for name in _fast_tier_harnesses() if name not in kani_target
    ]
    assert not missing, (
        "the Makefile kani: target must run every fast-tier harness declared "
        f"in the MST harness module; missing: {missing!r}"
    )
