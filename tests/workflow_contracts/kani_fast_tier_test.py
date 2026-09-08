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

#: The first ``kani:`` target header and its immediately following
#: tab-indented recipe lines; the capture stops at the first line that is
#: not tab-indented.
KANI_TARGET_RE = re.compile(
    r"^kani:[^\n]*\n(?P<recipe>(?:\t[^\n]*(?:\n|$))*)",
    re.MULTILINE,
)

#: An executable Kani command in the ``kani:`` recipe. The optional leading
#: variable expansions cover the environment prefix; comments and shell text
#: cannot satisfy this pattern.
KANI_COMMAND_RE = re.compile(
    r"^\t@?(?:\$\([A-Z_]+\)\s+)*\$\(CARGO\)\s+kani\b(?P<arguments>.*)$"
)
HARNESS_ARGUMENT_RE = re.compile(r"(?:^|\s)--harness\s+(verify_\w+)(?:\s|$)")


def _fast_tier_harnesses() -> list[str]:
    """Return every proof name declared in the fast-tier harness modules."""
    names: list[str] = []
    for source in FAST_TIER_SOURCES:
        names.extend(PROOF_RE.findall(source.read_text(encoding="utf-8")))
    return names


def _executed_harnesses(kani_target: str) -> set[str]:
    """Return harnesses passed to executable Kani commands in one recipe."""
    harnesses: set[str] = set()
    for line in kani_target.splitlines():
        match = KANI_COMMAND_RE.match(line)
        if match:
            harnesses.update(HARNESS_ARGUMENT_RE.findall(match["arguments"]))
    return harnesses


@pytest.mark.parametrize(
    "recipe",
    (
        "\t# $(CARGO) kani --harness verify_mst_structural_correctness_4_nodes",
        "\techo $(CARGO) kani --harness verify_mst_structural_correctness_4_nodes",
        "\tprintf '%s\\n' '--harness verify_mst_structural_correctness_4_nodes'",
    ),
)
def test_executed_harnesses_rejects_nonexecuting_recipe_text(recipe: str) -> None:
    """Proof names in comments or output commands cannot satisfy the contract."""
    assert not _executed_harnesses(recipe)


@pytest.fixture(scope="module")
def kani_target() -> str:
    """Return the recipe body of the Makefile's ``kani:`` target."""
    match = KANI_TARGET_RE.search(MAKEFILE_PATH.read_text(encoding="utf-8"))
    assert match and match["recipe"], (
        "the Makefile must define a kani: target with a recipe"
    )
    return match["recipe"].rstrip("\n")


def test_fast_tier_sources_declare_harnesses() -> None:
    """Guard the derivation: an empty scan would pass everything vacuously."""
    harnesses = _fast_tier_harnesses()
    assert harnesses, (
        f"no proofs found in {[str(p) for p in FAST_TIER_SOURCES]}; "
        "the derivation regex or source list is broken"
    )


def test_make_kani_runs_every_fast_tier_harness(kani_target: str) -> None:
    """Each documented fast-tier proof is an executed Kani harness argument."""
    executed_harnesses = _executed_harnesses(kani_target)
    missing = [
        name for name in _fast_tier_harnesses() if name not in executed_harnesses
    ]
    assert not missing, (
        "the Makefile kani: target must run every fast-tier harness declared "
        f"in the MST harness module; missing: {missing!r}"
    )
