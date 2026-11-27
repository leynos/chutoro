"""Drive the compile-contract test sweep over trees built to separate its rules.

The sweep in :mod:`rust_test_discovery` answers which files it visits, whether
an unreadable source fails instead of shrinking the answer, and whether the
reading is per test function rather than per file. Over this repository's own
sources the sweep agrees with itself whatever it does, because the set it is
compared against was written from what it found. These tests build trees and
sources that state each decision separately.

The override contract that consumes the sweep lives in
:mod:`trybuild_override_test`.

Run via ``make test-workflow-contracts``.
"""

import pathlib

import pytest
from rust_source_fixtures import (
    BRACE_IN_A_LITERAL,
    CHAR_LITERAL_BRACE,
    COMMENTED_ATTRIBUTE,
    GENERIC_SIGNATURE,
    HELPER_BELOW_A_TEST,
    LIFETIME,
    LIFETIME_THEN_CHAR,
    NESTED_ASYNC,
    QUALIFIED,
    RAW_STRING_WITH_A_QUOTE,
    RETRIES_ONLY,
    RSTEST,
    SIBLINGS,
    SPAWNER,
    SPLIT_ATTRIBUTE,
    SPLIT_CASE_ATTRIBUTE,
    SPACED_ATTRIBUTE,
)
from rust_test_discovery import compile_contract_tests, discovered_tests


def test_the_sweep_walks_a_tree_it_is_given(tmp_path: pathlib.Path) -> None:
    """Which files the sweep visits, driven over a tree built to separate them.

    Over this repository's own sources the sweep agrees with itself whatever
    it visits, because the set it is compared against was written from what it
    found. This tree states the three decisions separately: a source below
    `tests/` is read, one elsewhere in the tree is not, and one under a
    `target` directory is not even when its path contains `tests`, nor is one
    under the repository-local uv cache or tool directory, which hold other
    repositories' checkouts.

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
    # A differently named spawner under each uv directory, so a sweep that read
    # either would add a key rather than overwrite the one the assertion expects.
    for directory in (".uv-cache", ".uv-tools"):
        foreign = (
            tmp_path / directory / "git-v0" / "checkouts" / "abc" / "toy" / "tests"
        )
        foreign.mkdir(parents=True)
        name = "a_foreign_build_in_" + directory.lstrip(".").replace("-", "_")
        (foreign / "spawner.rs").write_text(
            SPAWNER.replace("checks_the_fixture_crate", name), encoding="utf-8"
        )

    assert discovered_tests(tmp_path) == {
        "checks_the_fixture_crate": "tests/harness.rs"
    }, (
        "the sweep must read every Rust source below a `tests/` directory of "
        "the given root, skip build output under `target` and the uv directories, "
        "and report each "
        "path relative to that root"
    )


@pytest.mark.parametrize(
    ("make_unreadable", "error"),
    [
        pytest.param(
            lambda path: path.write_bytes(b"fn \xff() {}\n"),
            UnicodeDecodeError,
            id="invalid-utf-8",
        ),
        pytest.param(lambda path: path.mkdir(), IsADirectoryError, id="a-source-read-failure"),
    ],
)
def test_an_unreadable_source_fails_the_sweep_rather_than_shrinking_it(
    tmp_path: pathlib.Path,
    make_unreadable: typ.Callable[[pathlib.Path], object],
    error: type[Exception],
) -> None:
    """The documented promise: raise, never return a smaller set.

    A readable costly test sits beside the bad file, so a sweep that skipped
    the failure would return it alone, which is exactly the partial answer
    that lets a test lose its override without anything failing.
    """
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "harness.rs").write_text(SPAWNER, encoding="utf-8")
    make_unreadable(tmp_path / "tests" / "broken.rs")
    with pytest.raises(error):
        discovered_tests(tmp_path)


def test_a_root_below_a_target_directory_is_still_swept(tmp_path: pathlib.Path) -> None:
    """Build output is excluded below the root, not anywhere in its path.

    A `cargo package` verification tree sits under `target/package/`, and a
    filter over the absolute path would drop every file there, so discovery
    would return nothing and the coverage rule would pass with nothing to
    check.
    """
    root = tmp_path / "target" / "package" / "crate"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "harness.rs").write_text(SPAWNER, encoding="utf-8")
    assert discovered_tests(root) == {"checks_the_fixture_crate": "tests/harness.rs"}


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
            id="a-comment-between-two-attributes",
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
