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

import math
import pathlib
import re
import tomllib
import typing as typ

import pytest
from nextest_durations import NextestDurationError, _seconds
from rust_source_fixtures import (
    EXTENDING,
    BRACE_IN_A_LITERAL,
    CHAR_LITERAL_BRACE,
    COMMENTED_ATTRIBUTE,
    LIFETIME,
    LIFETIME_THEN_CHAR,
    GENERIC_SIGNATURE,
    HELPER_BELOW_A_TEST,
    SPLIT_ATTRIBUTE,
    SPLIT_CASE_ATTRIBUTE,
    SPACED_ATTRIBUTE,
    NESTED_ASYNC,
    QUALIFIED,
    RAW_STRING_WITH_A_QUOTE,
    RETRIES_ONLY,
    RSTEST,
    SIBLINGS,
    SPAWNER,
)
from rust_source_reading import compile_contract_tests, discovered_tests

from workflow_support import ROOT as REPO_ROOT

NEXTEST_CONFIG = REPO_ROOT / ".config" / "nextest.toml"

#: Every test in this repository that pays a nested build's cost. Pinned as
#: well as discovered: the rule below is satisfied by a discovery that finds
#: nothing, so without this a reading narrowed by accident would read as a
#: repository with no such tests. Two of these construct a trybuild harness in
#: one file, one spawns `cargo check` against a fixture manifest, and the next
#: two are single-test files. The last three are the clustering-result API
#: checks: two trybuild harnesses gated on either side of the `cpu` feature,
#: and the one that runs the CPU-disabled side through a nested `cargo test`.
COMPILE_CONTRACT_TESTS: typ.Final[frozenset[str]] = frozenset({
    "session_api_compiles_when_cpu_feature_is_enabled",
    "session_api_is_unavailable_without_cpu_feature",
    "dataset_recipe_phase_order",
    "arrow_parquet_types_share_one_family",
    "portable_simd_gating_compile_checks",
    "clustering_result_panicking_constructor_is_private_when_cpu_enabled",
    "clustering_result_panicking_constructor_is_unavailable_without_cpu",
    "clustering_result_api_is_checked_without_cpu",
})

def _period_seconds(timeout: object) -> float | None:
    """Return a `slow-timeout`'s terminating allowance in seconds, or None.

    A timeout without `terminate-after` kills nothing: it marks a test slow and
    lets it run for ever, so it is not an allowance and reads as absent here.
    The allowance is the period times the number of periods, which is what
    nextest waits before terminating.

    Returns
    -------
    float or None
        The seconds a matched test may run, or None when nothing terminates.
    """
    if not isinstance(timeout, dict) or "terminate-after" not in timeout:
        return None
    # nextest reads periods with humantime, so `5m`, `2h 37min` and `1m30s`
    # are all valid and a seconds-only reader answered None for each. That
    # dropped the override from the extending set and reported the tests it
    # covers as uncovered, on a configuration nextest accepts. One reader owns
    # nextest durations in this repository; a second would be the same defect
    # as the list this contract exists to replace.
    try:
        period = _seconds(str(timeout.get("period", "")))
    except NextestDurationError:
        return None
    return period * float(timeout["terminate-after"])


def _base_allowance(profile: dict[str, object], default: dict[str, object]) -> float:
    """Return the allowance a test of this profile gets with no override.

    A profile that declares no `slow-timeout` inherits the default profile's,
    which is how `ci` is bounded here.

    Returns
    -------
    float
        The base allowance in seconds, or infinity when nothing terminates.
    """
    own = _period_seconds(profile.get("slow-timeout"))
    inherited = _period_seconds(default.get("slow-timeout"))
    found = own if own is not None else inherited
    return math.inf if found is None else found


def _extending_filters(
    config: dict[str, typ.Any] | None = None,
) -> dict[str, list[str]]:
    """Return each profile's filters that actually lengthen the allowance.

    An override supplies only the settings it declares to the tests its filter
    matches, so an override that sets `retries` or `threads-required` and no
    timeout leaves a matched test on the base allowance. Collecting every
    filter would count such an override as cover, and would equally miss an
    override whose timeout was lowered to the base.

    Kept per profile rather than pooled. Pooling lets a test named on one
    profile satisfy the rule for every profile, which is the shape the estate
    keeps finding: a `ci` profile that runs the expensive tests while the
    allowance sized for them was written only under `default`.

    Parameters
    ----------
    config
        A parsed nextest configuration. This repository's own is read when
        none is given, so the reading can be driven with configurations the
        tree does not contain: over its own, every override that names a
        compile-contract test also extends its allowance, so nothing here
        distinguishes a reading that checks for that from one that does not.

    Returns
    -------
    dict[str, list[str]]
        Profile name to the filter of each override that grants more time than
        the profile's base allowance.
    """
    if config is None:
        config = tomllib.loads(NEXTEST_CONFIG.read_text(encoding="utf-8"))
    profiles = config.get("profile", {})
    default = profiles.get("default", {})
    return {
        name: [
            str(override.get("filter", ""))
            for override in profile.get("overrides", [])
            if (granted := _period_seconds(override.get("slow-timeout"))) is not None
            and granted > _base_allowance(profile, default)
        ]
        for name, profile in profiles.items()
    }


#: An operator that removes tests from a filterset rather than adding them.
#: `not test(/costly/)` and `all() - test(/costly/)` both name `costly` and
#: both exclude it, so a reader that looks for the name in the text reports
#: coverage exactly where nextest applies none.
_EXCLUDING_OPERATOR = re.compile(r"(?:^|[^\w])(?:not\b|!|-)")


def _names_test(filter_text: str, name: str) -> bool:
    """Return whether a filter names this test rather than one containing it.

    A substring test is not enough. `dataset_recipe` is a substring of
    `dataset_recipe_phase_order`, so a filter naming the longer test would
    report the shorter one as covered and leave it on the base allowance.
    Identifier boundaries are required on both sides.

    A filter carrying an excluding operator names nothing here. Evaluating a
    filterset is nextest's work, and a reader that guessed at set arithmetic
    would be the same defect one layer down; failing closed costs an override
    that must be written as a positive union, and says so.

    Examples
    --------
    >>> _names_test("test(/a_b_c|d_e/)", "a_b_c")
    True
    >>> _names_test("test(/a_b_c/)", "a_b")
    False
    >>> _names_test("not test(/a_b_c/)", "a_b_c")
    False
    >>> _names_test("all() - test(/a_b_c/)", "a_b_c")
    False
    """
    if _EXCLUDING_OPERATOR.search(filter_text):
        return False
    return re.search(rf"(?<![0-9A-Za-z_]){re.escape(name)}(?![0-9A-Za-z_])", filter_text) is not None


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
    by_profile = _extending_filters()
    uncovered = {
        f"{profile}:{name}": path
        for profile, filters in by_profile.items()
        for name, path in discovered_tests().items()
        if not any(_names_test(filter_text, name) for filter_text in filters)
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
    assert set(discovered_tests()) == COMPILE_CONTRACT_TESTS, (
        f"the discovery must find exactly {sorted(COMPILE_CONTRACT_TESTS)}; "
        f"it found {sorted(discovered_tests())}. A test that has gained or lost a "
        f"nested build belongs in this set, and so does its allowance"
    )


def test_the_sweep_walks_a_tree_it_is_given(tmp_path: pathlib.Path) -> None:
    """Which files the sweep visits, driven over a tree built to separate them.

    Over this repository's own sources the sweep agrees with itself whatever
    it visits, because the set it is compared against was written from what it
    found. This tree states the three decisions separately: a source below
    `tests/` is read, one elsewhere in the tree is not, and one under a
    `target` directory is not even when its path contains `tests`.

    The paths are relative to the given root, not to this repository, which is
    what makes the reported path usable by a caller sweeping anything else.
    """
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "harness.rs").write_text(SPAWNER, encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.rs").write_text(SIBLINGS, encoding="utf-8")
    stale = tmp_path / "target" / "debug" / "tests"
    stale.mkdir(parents=True)
    (stale / "old.rs").write_text(SIBLINGS, encoding="utf-8")

    assert discovered_tests(tmp_path) == {
        "checks_the_fixture_crate": "tests/harness.rs"
    }, (
        "the sweep must read every Rust source below a `tests/` directory of "
        "the given root, skip build output under `target`, and report each "
        "path relative to that root"
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(SIBLINGS, ["compiles_the_fixture"], id="one-of-two-siblings"),
        pytest.param(SPAWNER, ["checks_the_fixture_crate"], id="a-nested-cargo-spawn"),
        # This repository declares tests with `rstest` as a matter of course. A
        # costly test written that way was discovered by nothing, and the
        # pinned set below would not have moved, so the missing override would
        # have passed.
        pytest.param(
            RSTEST, ["checks_with_the_repository_attribute"], id="an-rstest-attribute"
        ),
        # Four ways a literal can end a body early or hide a marker. Each
        # under-reports, which is the direction nothing else catches: the test
        # drops out of discovery rather than failing anything.
        pytest.param(
            BRACE_IN_A_LITERAL, ["writes_a_brace_then_builds"], id="a-brace-in-a-string"
        ),
        pytest.param(
            RAW_STRING_WITH_A_QUOTE,
            ["documents_a_path_then_builds"],
            id="a-quote-and-a-slash-pair-inside-a-raw-string",
        ),
        pytest.param(
            CHAR_LITERAL_BRACE,
            ["closes_on_a_char_then_builds"],
            id="a-brace-in-a-character-literal",
        ),
        pytest.param(
            LIFETIME, ["builds_the_fixture"], id="a-lifetime-is-not-a-literal"
        ),
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
        pytest.param(
            NESTED_ASYNC,
            ["builds_the_fixture_crate_V2"],
            id="an-indented-async-test-with-an-uppercase-name",
        ),
        pytest.param(
            QUALIFIED,
            ["spawns_a_nested_cargo"],
            id="a-declaration-behind-every-qualifier",
        ),
        pytest.param(
            LIFETIME_THEN_CHAR,
            ["borrows_then_builds"],
            id="a-lifetime-and-a-later-character-literal",
        ),
        pytest.param(
            SPACED_ATTRIBUTE,
            ["builds_under_a_spaced_attribute"],
            id="an-attribute-spaced-around-its-path-separator",
        ),
        pytest.param(
            GENERIC_SIGNATURE,
            ["builds_for_any_input"],
            id="a-nested-generic-signature",
        ),
        pytest.param(
            COMMENTED_ATTRIBUTE,
            ["builds_after_a_comment"],
            id="a-comment-between-the-attribute-and-the-signature",
        ),
        pytest.param(
            HELPER_BELOW_A_TEST,
            ["checks_the_fixture"],
            id="a-costly-helper-below-a-test",
        ),
        pytest.param(
            SPLIT_ATTRIBUTE,
            ["builds_under_a_split_attribute"],
            id="an-attribute-spanning-lines",
        ),
        pytest.param(
            SPLIT_CASE_ATTRIBUTE,
            ["builds_for_a_split_case"],
            id="a-case-attribute-spanning-lines",
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


@pytest.mark.parametrize(
    ("filter_text", "name", "names_it"),
    [
        pytest.param("test(/a_b_c|d_e/)", "a_b_c", True, id="one-alternative-of-many"),
        pytest.param("test(/a_b_c/)", "a_b_c", True, id="the-only-alternative"),
        # The case codex raised: a shorter name is a substring of a longer one,
        # so a filter naming the longer test would report the shorter as
        # covered and leave it on the base allowance.
        pytest.param("test(/a_b_c/)", "a_b", False, id="a-prefix-of-another-name"),
        pytest.param("test(/a_b_c/)", "b_c", False, id="a-suffix-of-another-name"),
        pytest.param("test(/other/)", "a_b_c", False, id="a-different-name"),
        # An excluding operator names the test and removes it, which a reader
        # that searches the text reports as coverage with the sign inverted.
        pytest.param("not test(/a_b_c/)", "a_b_c", False, id="a-negated-filter"),
        pytest.param("!test(/a_b_c/)", "a_b_c", False, id="a-negation-in-symbols"),
        pytest.param(
            "all() - test(/a_b_c/)", "a_b_c", False, id="a-difference-filterset"
        ),
        pytest.param(
            "test(/a_b_c/) & not test(/d_e/)",
            "a_b_c",
            False,
            id="a-negation-anywhere-in-the-expression",
        ),
        # A hyphen inside an identifier is not an operator: `chutoro-benches`
        # is a package name and this repository's own filters carry it.
        pytest.param(
            "package(chutoro-benches) & test(/a_b_c/)",
            "a_b_c",
            True,
            id="a-hyphen-inside-a-package-name",
        ),
    ],
)
def test_a_filter_names_a_test_only_at_identifier_boundaries(
    filter_text: str, name: str, *, names_it: bool
) -> None:
    """Coverage is a name match, and a substring is not one.

    Every test in this repository has a name no other contains, so over its own
    configuration a substring test and a boundary test agree exactly. These
    cases are what separate them.
    """
    assert _names_test(filter_text, name) is names_it, (
        f"`{filter_text}` must {'name' if names_it else 'not name'} `{name}`"
    )


@pytest.mark.parametrize(
    ("timeout", "seconds"),
    [
        pytest.param(
            {"period": "300s", "terminate-after": 1}, 300.0, id="one-period"
        ),
        pytest.param(
            {"period": "60s", "terminate-after": 5}, 300.0, id="five-periods"
        ),
        # A timeout without `terminate-after` kills nothing: it marks a test
        # slow and lets it run for ever. It is not an allowance, and an
        # override carrying one grants no more time than the base.
        pytest.param({"period": "900s"}, None, id="nothing-terminates"),
        pytest.param(None, None, id="no-timeout-at-all"),
        pytest.param("300s", None, id="a-bare-period-string"),
        # nextest parses periods with humantime, which reads a sequence of
        # value-and-unit pairs and sums them. A seconds-only reader answered
        # None for each of these, dropped the override from the extending set,
        # and reported the tests it covers as uncovered on a configuration
        # nextest accepts.
        pytest.param({"period": "5m", "terminate-after": 1}, 300.0, id="minutes"),
        pytest.param({"period": "1m30s", "terminate-after": 1}, 90.0, id="two-pairs"),
        pytest.param(
            {"period": "2h 37min", "terminate-after": 1}, 9420.0, id="spaced-pairs"
        ),
        pytest.param({"period": "300ms", "terminate-after": 1}, 0.3, id="milliseconds"),
        # Anything humantime refuses has no allowance to compare, so it reads
        # as absent rather than as zero.
        pytest.param({"period": "soon", "terminate-after": 1}, None, id="not-a-period"),
        pytest.param({"period": "", "terminate-after": 1}, None, id="an-empty-period"),
    ],
)
def test_an_allowance_is_a_period_that_terminates(
    timeout: object, seconds: float | None
) -> None:
    """What counts as more time than the base, and what only looks like it.

    An override that sets `retries` or `threads-required` and no timeout leaves
    a matched test on the base allowance. Reading every filter as cover, as
    this contract first did, counts such an override and would equally miss an
    override whose timeout had been lowered back to the base.
    """
    assert _period_seconds(timeout) == seconds, (
        f"{timeout!r} must read as {seconds!r} seconds of terminating allowance"
    )


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(EXTENDING, ["test(/costly/)"], id="an-override-that-extends"),
        pytest.param(RETRIES_ONLY, ["test(/other/)"], id="an-override-that-does-not"),
    ],
)
def test_only_an_override_that_lengthens_the_allowance_counts_as_cover(
    config: dict[str, typ.Any], expected: list[str]
) -> None:
    """An override supplies only the settings it declares.

    A filter that matches a costly test while setting `retries` or
    `threads-required` and no timeout leaves that test on the base allowance,
    so collecting every filter would count it as cover. Over this repository's
    own configuration the two readings agree exactly, because every override
    naming a compile-contract test also extends its allowance; these
    configurations are what separate them.
    """
    assert _extending_filters(config)["default"] == expected, (
        f"only an override granting more than the base allowance is cover; "
        f"got {_extending_filters(config)['default']!r}"
    )
