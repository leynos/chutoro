"""Repository-wide contract for GitHub Actions pins.

Dependabot owns the pin values and updates them as a single group, so these
tests assert shape and consistency rather than any specific commit. Naming a
SHA here would duplicate the one thing that is expected to change, turning
every routine bump into a spurious failure.

Two properties are checked across every workflow:

- every third-party ``uses:`` reference is pinned to a full 40-hex commit
  SHA, not a branch or tag, so a moved tag cannot alter what CI executes; and
- an action referenced from more than one workflow resolves to the *same*
  SHA everywhere, so a partially applied bump is caught rather than leaving
  workflows silently running different revisions of the same action.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

#: ``owner/repo[/path]@ref`` split into the action identity and its ref.
USES_RE = re.compile(r"^(?P<action>[^@]+)@(?P<ref>.+)$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")

#: Local composite actions and reusable workflows referenced by relative path
#: carry no ref, so they are outside this contract.
LOCAL_PREFIXES = ("./", "../")


def _workflow_files() -> list[Path]:
    """Return every workflow definition in the repository."""
    return sorted(
        path
        for path in WORKFLOW_DIR.iterdir()
        if path.suffix in {".yml", ".yaml"} and path.is_file()
    )


def _iter_uses(document: object) -> list[str]:
    """Return every ``uses:`` value anywhere in a parsed workflow."""
    found: list[str] = []
    if isinstance(document, dict):
        for key, value in document.items():
            if key == "uses" and isinstance(value, str):
                found.append(value)
            else:
                found.extend(_iter_uses(value))
    elif isinstance(document, list):
        for item in document:
            found.extend(_iter_uses(item))
    return found


@pytest.fixture(scope="module")
def references() -> dict[str, dict[str, set[str]]]:
    """Map each action to the SHAs it is pinned at, and where.

    Shaped as ``{action: {sha: {workflow, ...}}}`` so a failure can name the
    workflows that disagree.
    """
    collected: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for path in _workflow_files():
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        for uses in _iter_uses(document):
            if uses.startswith(LOCAL_PREFIXES):
                continue
            match = USES_RE.match(uses)
            assert match, f"{path.name}: unparseable uses value {uses!r}"
            collected[match["action"]][match["ref"]].add(path.name)
    return collected


def test_workflows_are_discovered() -> None:
    """Guard the derivation: an empty scan would pass everything vacuously."""
    workflows = _workflow_files()
    assert workflows, f"no workflow definitions found under {WORKFLOW_DIR}"


def test_every_action_is_pinned_to_a_commit_sha(
    references: dict[str, dict[str, set[str]]],
) -> None:
    """No workflow may track a branch or a movable tag."""
    assert references, "no external action references were discovered"

    unpinned = {
        f"{action}@{ref}": sorted(workflows)
        for action, refs in references.items()
        for ref, workflows in refs.items()
        if not SHA_RE.match(ref)
    }
    assert not unpinned, (
        "every action must be pinned to a full 40-hex commit SHA rather than "
        f"a branch or tag; these are not: {unpinned!r}"
    )


def test_shared_actions_are_consistent_across_workflows(
    references: dict[str, dict[str, set[str]]],
) -> None:
    """An action used more than once resolves to one SHA everywhere.

    This is what catches a partially applied Dependabot bump: the pin itself
    may be any value, but it must not differ between workflows.
    """
    divergent = {
        action: {ref: sorted(workflows) for ref, workflows in refs.items()}
        for action, refs in references.items()
        if len(refs) > 1
    }
    assert not divergent, (
        "each action must resolve to the same commit SHA in every workflow; "
        f"these diverge: {divergent!r}"
    )
