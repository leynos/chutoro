"""Every compile-contract test has an allowance sized for the build it spawns.

A per-test ``terminate-after`` and a name-based override list are a pair that
rots apart. The list is written once against the names of the day and is never
re-derived, and neither a passing run nor a green gate notices a test that has
fallen out of it, because the cost only appears on a cold cache.

The cost is what defines the set, not the crate. A ``trybuild::TestCases``
harness compiles a scratch crate against this workspace's dependency graph; a
test that spawns ``cargo check`` against a fixture manifest pays the same price
by a different route. Both are discovered here, because an estate survey found
the second kind is where the list breaks: one test of a pair named in the
override and its sibling, in the same file, left on the base allowance.

The discovery is per test function rather than per file, for the same reason.
A file-level reading would report this repository as fully covered.

Run via ``make test-workflow-contracts``.
"""

import re
import tomllib
import typing as typ

import pytest

from workflow_support import ROOT as REPO_ROOT

NEXTEST_CONFIG = REPO_ROOT / ".config" / "nextest.toml"

#: Every test in this repository that pays a nested build's cost. Pinned as
#: well as discovered: the rule below is satisfied by a discovery that finds
#: nothing, so without this a reading narrowed by accident would read as a
#: repository with no such tests. Two of these construct a trybuild harness in
#: one file, one spawns `cargo check` against a fixture manifest, and the last
#: two are single-test files.
COMPILE_CONTRACT_TESTS: typ.Final[frozenset[str]] = frozenset({
    "session_api_compiles_when_cpu_feature_is_enabled",
    "session_api_is_unavailable_without_cpu_feature",
    "dataset_recipe_phase_order",
    "arrow_parquet_types_share_one_family",
    "portable_simd_gating_compile_checks",
})

#: Sources of the cost this contract is about. A constructed ``TestCases``
#: compiles a scratch crate; a spawned ``cargo`` builds a fixture workspace.
#: Naming the crate is not enough for either: an import that is never used, or
#: a paragraph explaining why a harness was removed, costs nothing.
COST_MARKERS: typ.Final[tuple[str, ...]] = (
    "TestCases::new(",
    'Command::new(env!("CARGO"))',
    'Command::new("cargo")',
)

#: Whitespace Rust permits around a path separator or before a call's
#: parenthesis, which the formatter removes. Normalised rather than depended
#: upon, so the reading does not rest on the formatter having run.
_SPACED_SYNTAX = re.compile(r"\s*(::)\s*|\s+(\()")

#: A test function's declaration. `#[test]` and its cfg attributes may sit
#: between the attribute and the signature, so the name is taken from the
#: signature and the attribute is found by walking backwards.
_FUNCTION = re.compile(r"^fn (?P<name>[a-z_][a-z0-9_]*)\s*\(", re.MULTILINE)


def _block_comment(text: str, index: int) -> tuple[str, int]:
    """Return blanks for a block comment at `index`, and the position after it.

    Nested block comments are counted, because Rust permits them and a scan
    that stopped at the first `*/` would resume inside a comment.
    """
    depth = 0
    position = index
    while position < len(text):
        pair = text[position : position + 2]
        depth += {"/*": 1, "*/": -1}.get(pair, 0)
        if pair in {"/*", "*/"}:
            position += 2
            if depth == 0:
                return _blanked(text[index:position]), position
            continue
        position += 1
    return _blanked(text[index:]), len(text)


def _line_comment(text: str, index: int) -> tuple[str, int]:
    """Return blanks for a line comment at `index`, and the position after it."""
    end = text.find("\n", index)
    end = len(text) if end == -1 else end
    return " " * (end - index), end


def _string_literal(text: str, index: int) -> tuple[str, int]:
    """Return a string literal unchanged, and the position after it.

    Kept rather than blanked: a `//` inside a path argument is not a comment,
    and blanking the rest of that line would hide a call written after it.
    """
    position = index + 1
    while position < len(text) and text[position] != '"':
        position += 2 if text[position] == "\\" else 1
    return text[index : position + 1], position + 1


def _blanked(text: str) -> str:
    """Return the text with every character but a newline replaced by a space.

    Positions are preserved so a body's extent is unchanged by blanking.
    """
    return "".join(" " if character != "\n" else "\n" for character in text)


#: Each opener, and the scanner that consumes what it opens.
_SCANNERS: typ.Final[tuple[tuple[str, typ.Any], ...]] = (
    ("/*", _block_comment),
    ("//", _line_comment),
    ('"', _string_literal),
)


def _without_comments(text: str) -> str:
    """Return the source with line and block comments replaced by spaces.

    A marker inside a comment is a mention, not a cost. Written as a scan
    rather than a pattern because it has to know when it is inside a string
    literal.

    A raw string literal's hashes are not handled; this repository has none in
    a test body, and the failure mode is over-reporting, which the pinned set
    above catches.

    Returns
    -------
    str
        The source with every comment blanked, positions otherwise preserved.
    """
    out = []
    index = 0
    while index < len(text):
        for opener, scan in _SCANNERS:
            if text.startswith(opener, index):
                emitted, index = scan(text, index)
                out.append(emitted)
                break
        else:
            out.append(text[index])
            index += 1
    return "".join(out)


def _normalised(text: str) -> str:
    """Return the source with comments blanked and spacing normalised."""
    without = _without_comments(text)
    return _SPACED_SYNTAX.sub(lambda found: found.group(1) or found.group(2), without)


def _body(text: str, start: int) -> str:
    """Return one function's body, by matching braces from its signature.

    Returns
    -------
    str
        The text between the function's opening and closing brace, or the
        remainder of the file when the braces do not balance, which reads as
        a larger body and so can only over-report.
    """
    opening = text.find("{", start)
    if opening == -1:
        return ""
    depth = 0
    for index, character in enumerate(text[opening:], opening):
        depth += {"{": 1, "}": -1}.get(character, 0)
        if depth == 0:
            return text[opening : index + 1]
    return text[opening:]


def _is_test(text: str, start: int) -> bool:
    """Return whether the declaration at `start` carries a `#[test]` attribute.

    Attributes are read backwards from the signature, because `#[cfg(...)]`
    and doc comments may sit between the two.
    """
    preceding = text[:start].rstrip().split("\n")
    for line in reversed(preceding):
        stripped = line.strip()
        if stripped.startswith("#[test]"):
            return True
        if not stripped.startswith(("#[", "///", "//", "//!")):
            return False
    return False


def compile_contract_tests(text: str) -> list[str]:
    """Return every test in the source that pays a nested build's cost.

    Examples
    --------
    >>> compile_contract_tests(
    ...     "#[test]\\nfn a() { let c = trybuild::TestCases::new(); }"
    ... )
    ['a']
    >>> compile_contract_tests("#[test]\\nfn b() { assert!(true); }")
    []
    >>> compile_contract_tests("fn c() { let c = TestCases::new(); }")
    []
    """
    source = _normalised(text)
    found = []
    for match in _FUNCTION.finditer(source):
        if not _is_test(source, match.start()):
            continue
        body = _body(source, match.end())
        if any(marker in body for marker in COST_MARKERS):
            found.append(match["name"])
    return found


def _discovered() -> dict[str, str]:
    """Return each compile-contract test, mapped to the file declaring it."""
    return {
        name: path.relative_to(REPO_ROOT).as_posix()
        for path in sorted(REPO_ROOT.rglob("tests/**/*.rs"))
        if "target" not in path.parts
        for name in compile_contract_tests(path.read_text(encoding="utf-8"))
    }


def _override_filters() -> dict[str, list[str]]:
    """Return each profile's override filters, keyed by profile name.

    Kept per profile rather than pooled. Pooling them lets a test named on
    one profile satisfy the rule for every profile, which is the shape the
    estate keeps finding: a `ci` profile that runs the expensive tests while
    the allowance sized for them was written only under `default`.

    Returns
    -------
    dict[str, list[str]]
        Profile name to the filter of each override it declares.
    """
    config = tomllib.loads(NEXTEST_CONFIG.read_text(encoding="utf-8"))
    return {
        name: [str(override.get("filter", "")) for override in profile.get("overrides", [])]
        for name, profile in config.get("profile", {}).items()
    }


def test_the_base_allowance_terminates() -> None:
    """The premise this contract rests on, asserted rather than assumed.

    Without `terminate-after` nothing is killed, so an uncovered test would
    merely be reported slow and this rule would guard a hazard that does not
    exist. If it ever fails, the rule needs rewriting rather than relaxing.
    """
    config = tomllib.loads(NEXTEST_CONFIG.read_text(encoding="utf-8"))
    timeout = config["profile"]["default"]["slow-timeout"]
    assert "terminate-after" in timeout, (
        "[profile.default] must declare slow-timeout with terminate-after; "
        "this contract exists because an uncovered test inherits it"
    )


def test_every_compile_contract_test_is_named_in_an_override() -> None:
    """The set is discovered from the tree, not listed here.

    Listing it would be the same defect one level up: a list written once
    against the names of the day. Discovery fails when a test is added, which
    is the moment its allowance has to be decided.
    """
    by_profile = _override_filters()
    uncovered = {
        f"{profile}:{name}": path
        for profile, filters in by_profile.items()
        for name, path in _discovered().items()
        if not any(name in filter_text for filter_text in filters)
    }
    assert not uncovered, (
        f"these tests spawn a build and inherit the base allowance: "
        f"{uncovered}. Each compiles against this workspace's dependency "
        f"graph and will be terminated on a cold run. Name each in the "
        f"trybuild override of every profile"
    )


def test_the_discovered_set_is_the_one_this_repository_has() -> None:
    """Pin what the discovery finds, not merely that it finds nothing extra.

    The rule above is satisfied by a discovery that reads nothing, because an
    empty set is covered. That is how a reading narrowed by accident goes
    unnoticed: drop the nested-cargo marker and the test this contract was
    written to catch stops being discovered rather than starting to fail.
    """
    assert set(_discovered()) == COMPILE_CONTRACT_TESTS, (
        f"the discovery must find exactly {sorted(COMPILE_CONTRACT_TESTS)}; "
        f"it found {sorted(_discovered())}. A test that has gained or lost a "
        f"nested build belongs in this set, and so does its allowance"
    )


#: Sources whose discovered set is the point, written out so the reading is
#: driven rather than inferred from a tree that happens to agree with it.
_SIBLINGS = """
#[test]
fn compiles_the_fixture() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/ok.rs");
}

#[test]
fn asserts_a_value() {
    assert_eq!(2 + 2, 4);
}
"""

_SPAWNER = """
#[test]
fn checks_the_fixture_crate() {
    let output = Command::new(env!("CARGO"))
        .arg("check")
        .output()
        .expect("cargo must run");
}
"""


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(_SIBLINGS, ["compiles_the_fixture"], id="one-of-two-siblings"),
        pytest.param(_SPAWNER, ["checks_the_fixture_crate"], id="a-nested-cargo-spawn"),
        pytest.param(
            '#[test]\nfn mentions_it() { /* trybuild::TestCases::new() */ }',
            [],
            id="named-in-a-comment",
        ),
        pytest.param(
            'fn helper() { let c = trybuild::TestCases::new(); }',
            [],
            id="a-helper-that-is-not-a-test",
        ),
        pytest.param(
            '#[test]\n#[cfg(feature = "cpu")]\nfn gated() { TestCases::new(); }',
            ["gated"],
            id="a-test-behind-a-cfg-attribute",
        ),
        pytest.param(
            '#[test]\nfn spaced() { let c = trybuild :: TestCases :: new (); }',
            ["spaced"],
            id="generously-spaced",
        ),
    ],
)
def test_the_discovery_reads_each_function_separately(
    source: str, expected: list[str]
) -> None:
    """A file-level reading would report this repository as fully covered.

    The defect this contract exists to catch is a sibling: two tests in one
    file, one named in the override and one not. A reading that asked whether
    the file pays the cost would say yes for both and find nothing, and over
    this repository's own sources it would agree with a correct reading
    exactly. These cases are what separate the two.

    The comment case matters for the same reason from the other side. A text
    match would report a test that only names the crate, and an over-reporting
    discovery is satisfied by naming an allowance for a test that does not
    need one, which is how a list grows entries nobody can justify.
    """
    assert compile_contract_tests(source) == expected, (
        f"the discovery must read {expected!r} from this source; it read "
        f"{compile_contract_tests(source)!r}"
    )
