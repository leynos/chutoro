"""Contract tests for the repository's CodeScene rule overrides.

`.codescene/code-health-rules.json` pointed at
`chutoro-providers/dense/src/simd/kernels.rs` for over a week after commit
7512003 moved that module root to `kernels/mod.rs`, so the only rule set the
repository has matched nothing and its Primitive Obsession waiver quietly
stopped applying (#253).

Nothing caught it. The file is valid JSON, `cs rules-config validate` accepts
it, and CodeScene's verdicts simply carry on without an override that matches
no file. The CodeScene CLI is not installed on the CI runners either, so these
tests hold the line where the gates actually run: they assert the documented
schema rather than merely that the file parses, and they assert that each
exemption still matches a tracked file, because a path that has gone stale is
an exemption that silently stops applying.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import json
import subprocess
import typing as typ
from pathlib import Path

import pytest
from workflow_support import ROOT

RULES_PATH = ROOT / ".codescene" / "code-health-rules.json"

#: Keys `cs docs code-health-rules-template` emits for a rule set. The `_doc`
#: suffixes are documentation the template carries itself, so they are
#: permitted rather than required by the schema check; a separate test
#: requires the one that justifies the exemption.
RULE_SET_KEYS = frozenset({
    "matching_content_path",
    "matching_content_path_doc",
    "content_filter",
    "rules",
    "thresholds",
})


def _tracked_files() -> frozenset[str]:
    """Return every path git tracks, as repository-relative POSIX strings.

    Matching against the index rather than the working tree keeps the check
    honest: an exemption satisfied only by an untracked or ignored file is an
    exemption no reviewer or CI run would ever see.
    """
    listing = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        capture_output=True,
        check=True,
        text=True,
    )
    return frozenset(entry for entry in listing.stdout.split("\0") if entry)


def _rule_sets() -> list[dict[str, typ.Any]]:
    """Return the file's rule sets, failing if the top-level shape is wrong."""
    document = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict), "the rule file must be a JSON object"
    rule_sets = document.get("rule_sets")
    assert isinstance(rule_sets, list), (
        "the rule file must carry a top-level 'rule_sets' array; CodeScene "
        "ignores a shape it does not recognize and reports nothing useful"
    )
    assert rule_sets, "an empty rule_sets array declares no override at all"
    return typ.cast("list[dict[str, typ.Any]]", rule_sets)


def test_the_rule_file_uses_the_documented_schema() -> None:
    """Every rule set and rule must use the keys and value types CodeScene reads.

    A key CodeScene does not recognize is dropped rather than rejected, so a
    typo fails exactly as a stale path does: silently, with the override
    ignored and the file still passing validation.
    """
    for rule_set in _rule_sets():
        assert isinstance(rule_set, dict), "each rule set must be an object"
        unexpected = set(rule_set) - RULE_SET_KEYS
        assert not unexpected, (
            f"unrecognized rule set keys {sorted(unexpected)}; CodeScene "
            "ignores what it does not recognize"
        )
        for rule in rule_set.get("rules", []):
            assert set(rule) == {"name", "weight"}, (
                f"a rule override carries exactly a name and a weight: {rule}"
            )
            assert isinstance(rule["name"], str) and " " in rule["name"], (
                "rules are named in prose, as in 'Primitive Obsession', not "
                f"as a hyphenated slug: {rule['name']!r}"
            )
            weight = rule["weight"]
            assert isinstance(weight, (int, float)) and 0.0 <= weight <= 1.0, (
                "a rule's weight is a relative multiplier between 0.0 and "
                f"1.0, not a threshold: {weight!r}"
            )


def test_every_exemption_is_justified() -> None:
    """Each rule set explains itself, so a reader can judge whether it still holds.

    An exemption without a stated reason is indistinguishable from one nobody
    revisited, which is how a narrow allowance becomes a permanent blind spot.
    """
    for rule_set in _rule_sets():
        justification = rule_set.get("matching_content_path_doc", "")
        assert isinstance(justification, str) and len(justification) > 80, (
            "each rule set needs a matching_content_path_doc saying why the "
            f"exemption is deliberate: {rule_set.get('matching_content_path')!r}"
        )


def test_every_exemption_still_matches_a_tracked_file() -> None:
    """A rule set's path must match at least one file git tracks.

    This is the assertion that fails on #253. A renamed or moved file leaves
    an exemption that reads as though it does something and does nothing.
    """
    tracked = _tracked_files()
    for rule_set in _rule_sets():
        pattern = rule_set.get("matching_content_path")
        assert isinstance(pattern, str) and pattern, (
            "each rule set must declare a matching_content_path"
        )
        matched = [
            found
            for found in ROOT.glob(pattern)
            if found.relative_to(ROOT).as_posix() in tracked
        ]
        assert matched, (
            f"no tracked file matches {pattern!r}; an exemption that matches "
            "nothing is either stale or was never right"
        )


@pytest.mark.parametrize(
    ("document", "failing_test"),
    [
        pytest.param(
            {
                "rule_sets": [
                    {
                        "matching_content_path": (
                            "chutoro-providers/dense/src/simd/kernels.rs"
                        ),
                        "matching_content_path_doc": (
                            "Kernel functions operate directly on `&[f32]` "
                            "slices and `usize` offsets because SIMD "
                            "intrinsics require contiguous, unboxed memory "
                            "and raw index arithmetic"
                        ),
                        "rules": [{"name": "Primitive Obsession", "weight": 0.0}],
                    }
                ]
            },
            "test_every_exemption_still_matches_a_tracked_file",
            id="the-stale-path-this-file-exists-to-catch",
        ),
        pytest.param(
            {"rule_sets": [{"rules": [{"name": "Primitive Obsession", "weight": 100}]}]},
            "test_the_rule_file_uses_the_documented_schema",
            id="a-threshold-where-a-weight-belongs",
        ),
        pytest.param(
            {"rule_sets": [{"rules": [{"name": "primitive-obsession", "weight": 0.0}]}]},
            "test_the_rule_file_uses_the_documented_schema",
            id="a-slug-where-a-prose-name-belongs",
        ),
        pytest.param(
            {"rules": {"primitive-obsession": {"threshold-by-pattern": {"**": 100}}}},
            "test_the_rule_file_uses_the_documented_schema",
            id="a-top-level-rules-object",
        ),
        pytest.param(
            {
                "rule_sets": [
                    {
                        "matching_content_path": (
                            "chutoro-providers/dense/src/simd/kernels/mod.rs"
                        ),
                        "matching_content_path_doc": "SIMD kernels",
                        "rules": [{"name": "Primitive Obsession", "weight": 0.0}],
                    }
                ]
            },
            "test_every_exemption_is_justified",
            id="a-justification-too-thin-to-judge",
        ),
    ],
)
def test_the_assertions_reject_the_shapes_they_exist_to_catch(
    document: dict[str, typ.Any],
    failing_test: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each check must fail on the defect it guards against.

    Without this the checks could assert nothing and still pass, which is the
    defect they exist to catch, one level up. The stale-path case is the exact
    content of the file before #253 was fixed.
    """
    path = tmp_path / "code-health-rules.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    # `raising` is left at its default on purpose: if this module is ever
    # imported under a different name the patch must fail loudly rather than
    # create a new attribute and let the test pass having patched nothing.
    monkeypatch.setattr(f"{__name__}.RULES_PATH", path)

    with pytest.raises(AssertionError):
        globals()[failing_test]()
