"""Contract tests for the HNSW scheduling-regression CI gate.

The two Rust regressions in ``hnsw::cpu::trim_tests`` are the runtime evidence
that trim scoring stays inside the insertion critical section. Evidence that
runs nowhere protects nothing: a renamed test, a removed Make target, or a CI
step deleted beside it would all leave the suite green.

These tests read the Makefile recipe and the ``build-test`` job in
``.github/workflows/ci.yml``. They pin the target, one invocation per fully
qualified exact-match test name, the zero-test failure flag on each, the
unconditional and blocking execution of the step within an unconditional job,
and the dependency on an installed ``cargo-nextest``. They also require default
features so the gate cannot become collateral damage when the coverage action's
feature set changes.

The reading itself, with the cases that keep its filtering honest, lives in
``scheduling_reading``; the Rust declaration reading is the suite's shared
``rust_declaration_reading``.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import pytest
from rust_declaration_reading import declares_test
from scheduling_invocation_reading import is_nextest_invocation
from scheduling_reading import (
    QUALIFIED_REGRESSIONS,
    REGRESSIONS,
    TARGET,
    feature_flags,
    invocations_missing_flag,
    makefile_text,
    nextest_invocations,
    phony_targets,
    recipe_commands,
    selected_regression,
    target_recipe,
)
from workflow_support import ROOT


def test_the_target_retires_when_the_filter_selects_nothing() -> None:
    """A filter that matches nothing must fail rather than pass.

    ``cargo nextest`` exits successfully when a filter selects no test. That
    default would let a rename retire the evidence silently, so the flag is
    part of the contract rather than a local convenience. Each invocation is
    checked on its own: counting flags across the whole recipe would accept a
    command that carries the flag twice beside one that omits it.
    """
    invocations = nextest_invocations()
    assert len(invocations) == len(REGRESSIONS), (
        "the target must run one nextest invocation per regression"
    )
    without_flag = invocations_missing_flag(invocations)
    assert not without_flag, (
        f"scheduling invocation(s) {without_flag} must pass --no-tests=fail"
    )


def test_a_flag_on_another_invocation_does_not_cover_a_missing_one() -> None:
    """The flag is a per-invocation requirement, not a recipe allowance.

    One command carrying the flag twice must not cover a second command that
    omits it; that is exactly the shape a whole-recipe count accepts.
    """
    invocations = [
        "cargo nextest run -E 'test(=a)' --no-tests=fail --no-tests=fail",
        "cargo nextest run -E 'test(=b)'",
    ]
    assert invocations_missing_flag(invocations) == [1], (
        "only the unflagged invocation may be reported"
    )


def test_a_flag_name_in_a_redirection_is_not_the_flag() -> None:
    """A redirection target spelling the flag is not the flag.

    ``nextest run --no-tests=pass >--no-tests=fail`` carries the opposite
    setting, so matching any word would let an empty selection pass the gate
    that exists to make it fail. The spaced form redirects to the same file,
    so it must not answer for the flag either.
    """
    invocations = [
        "cargo nextest run --no-tests=pass >--no-tests=fail",
        "cargo nextest run --no-tests=pass > --no-tests=fail",
        "cargo nextest run --no-tests=fail",
    ]
    assert invocations_missing_flag(invocations) == [0, 1], (
        "no redirection target may answer for the flag"
    )


@pytest.mark.parametrize(
    "command",
    [
        pytest.param(
            "echo $(CARGO) nextest run -p chutoro-core --lib",
            id="a-mentioned-command",
        ),
        pytest.param(
            "-$(CARGO) nextest run -p chutoro-core --lib",
            id="an-ignored-error",
        ),
        pytest.param(
            "./scripts/nextest-runner nextest run",
            id="an-unrelated-runner",
        ),
        pytest.param(
            "$(CARGO) nextest run -p chutoro-core --lib || true",
            id="an-or-true-suffix",
        ),
        pytest.param(
            "$(CARGO) nextest run -p chutoro-core --lib; true",
            id="a-semicolon-suffix",
        ),
        pytest.param(
            "$(CARGO) nextest run -p chutoro-core --lib ||true",
            id="an-unspaced-or-true-suffix",
        ),
        pytest.param(
            "$(CARGO) nextest run -p chutoro-core --lib"
            f" -E 'test(={QUALIFIED_REGRESSIONS[REGRESSIONS[0]]})'"
            " --no-tests=fail --no-run",
            id="a-compile-only-run",
        ),
    ],
)
def test_only_a_command_that_runs_nextest_counts(command: str) -> None:
    """A mention is not an invocation, and a mask hides a failure.

    Counting the text anywhere in a recipe accepts ``echo $(CARGO) nextest
    run ...``, which runs nothing while every argument assertion passes.
    Accepting a leading ``-``, a ``|| true`` suffix, or a command after ``;``
    is the same problem with a worse outcome: the regression runs, fails, and
    the target still succeeds. ``--no-run`` is the same problem again in a
    shape that satisfies every argument check: the command carries the exact
    selector and the zero-test flag, and compiles the tests without running
    them.
    """
    assert not is_nextest_invocation(command), (
        f"{command!r} must not count as a scheduling invocation"
    )


def test_the_same_command_without_no_run_is_the_invocation_it_must_be() -> None:
    """A compile-only rejection must not reject the run the target needs.

    The negative case carries the target's complete selector. The same
    command with the flag removed is exactly what the recipe runs, so a
    predicate that rejected it would leave the gate with no invocation at
    all.
    """
    command = (
        "$(CARGO) nextest run -p chutoro-core --lib"
        f" -E 'test(={QUALIFIED_REGRESSIONS[REGRESSIONS[0]]})'"
        " --no-tests=fail"
    )
    assert is_nextest_invocation(command), (
        f"{command!r} must count as a scheduling invocation"
    )


def test_the_recipe_does_not_ignore_errors() -> None:
    """Make suppresses a failure for any recipe line beginning with `-`.

    The CI step and job checks reject ``continue-on-error``, but neither sees
    a recipe-level prefix, so the target's own commands are read for one.
    """
    commands = recipe_commands(makefile_text(), TARGET)
    ignored = [command for command in commands if command.startswith("-")]
    assert not ignored, (
        f"the scheduling recipe must not ignore errors: {ignored}"
    )


def test_a_hash_is_a_comment_only_where_a_word_starts() -> None:
    """A ``#`` inside a word belongs to that word, not to a comment.

    Cutting at the first ``#`` turned
    ``nextest run >scheduling#log || true`` into an unmasked invocation, so a
    line that masks the regression's failure would have counted as the gate
    that runs it. A genuine trailing comment must still go, because an
    argument it mentioned would otherwise answer for an invocation that omits
    it. An escaped operator ends no word either, so ``>scheduling\\;#log``
    keeps its mask as well.
    """
    prefix = "$(CARGO) nextest run -p chutoro-core --lib"
    in_word = prefix + " >scheduling#log || true"
    escaped = prefix + " >scheduling\\;#log || true"
    commented = prefix + " # --no-tests=fail"
    assert recipe_commands(f"{TARGET}:\n\t{in_word}\n", TARGET) == [in_word], (
        "the hash inside the word must survive"
    )
    assert not is_nextest_invocation(in_word), (
        "the mask after the redirect must still disqualify the command"
    )
    assert recipe_commands(f"{TARGET}:\n\t{escaped}\n", TARGET) == [escaped], (
        "the escaped operator must keep the word whole"
    )
    assert not is_nextest_invocation(escaped), (
        "the mask after the escaped word must still disqualify the command"
    )
    assert recipe_commands(f"{TARGET}:\n\t{commented}\n", TARGET) == [prefix], (
        "the trailing comment must be removed"
    )


def test_each_regression_runs_in_its_own_invocation() -> None:
    """One combined filter cannot vouch for a single missing name.

    A disjunction still selects the surviving test when the other name stops
    matching, so ``--no-tests=fail`` never fires. Each invocation must select
    exactly one regression with an expression naming nothing else, and the
    invocations must cover every regression once: a per-name total spread
    across commands would accept two selectors in one command and an
    unrelated test in the other.
    """
    invocations = nextest_invocations()
    assert len(invocations) == len(REGRESSIONS), (
        "the target must run one nextest invocation per regression"
    )
    chosen = [selected_regression(command) for command in invocations]
    assert sorted(chosen) == sorted(REGRESSIONS), (
        "the invocations must cover each regression exactly once; "
        f"got {chosen}"
    )


def test_a_widened_expression_is_rejected() -> None:
    """A disjunction with all() keeps an invocation non-empty forever.

    Both invocations could carry ``test(=name) | all()`` and pass every count:
    the expression selects every test, so the zero-test flag never fires once
    the named regression is renamed. The whole expression has to name the
    regression, not merely contain its selector.
    """
    widened = (
        "cargo nextest run -E "
        f"'test(={QUALIFIED_REGRESSIONS[REGRESSIONS[0]]}) | all()'"
    )
    with pytest.raises(ValueError, match="must select exactly one regression"):
        selected_regression(widened)


def test_both_scheduling_regressions_are_selected() -> None:
    """Both test names must appear in the selection expressions.

    Dropping either name would leave the other's single observation standing
    in for a property that needs both a direct and a real-insertion witness.
    The substring form is rejected as well: nextest treats it as a partial
    match, so a renamed regression could leave a like-named fixture passing.
    """
    recipe = target_recipe(makefile_text(), TARGET)
    missing = [name for name in REGRESSIONS if name not in recipe]
    assert not missing, f"the filter must select {missing}"
    for name in REGRESSIONS:
        assert f"test({name})" not in recipe, (
            f"{name} must be matched exactly, not as a substring"
        )


def test_the_target_runs_the_core_library_with_default_features() -> None:
    """The gate is scoped to the core library and does not widen features.

    The regressions are feature-independent. ``--all-features`` would tie this
    evidence to the coverage action's configuration, so a feature change could
    alter what the scheduling gate builds.
    """
    recipe = target_recipe(makefile_text(), TARGET)
    assert "nextest run" in recipe, "the target must run nextest"
    assert "-p chutoro-core" in recipe, "the gate must select chutoro-core"
    assert "--lib" in recipe, "the gate must run the library target"
    flags = feature_flags(recipe)
    assert not flags, (
        f"the gate must build the default features; it passes {flags}"
    )


@pytest.mark.parametrize(
    "flag",
    [
        pytest.param("--all-features", id="all-features"),
        pytest.param("--no-default-features", id="no-default-features"),
        pytest.param("--features metrics", id="features"),
        pytest.param("--features=metrics", id="features-equals"),
        pytest.param("-F metrics", id="short-features"),
    ],
)
def test_every_feature_selection_spelling_is_rejected(flag: str) -> None:
    """A feature change must not be able to ride along with the gate.

    Rejecting ``--all-features`` alone leaves four other spellings that alter
    what the gate builds, including the ``--features=name`` form a check for
    the bare flag cannot see.
    """
    recipe = f"$(CARGO) nextest run -p chutoro-core --lib {flag}"
    assert feature_flags(recipe), f"{flag} must be rejected"


def test_the_target_is_phony() -> None:
    """A file named like the target must not satisfy it.

    A name that appears only in a Make comment declares nothing, so
    ``.PHONY: clean # test-scheduling-contract`` must not answer for the
    target either.
    """
    assert TARGET in phony_targets(makefile_text()), (
        f"{TARGET} must be declared phony"
    )
    commented = f".PHONY: clean {chr(35)} {TARGET}"
    assert TARGET not in phony_targets(commented), "comments declare nothing"


def test_the_target_is_declared_once() -> None:
    """A later duplicate recipe would override the one this contract reads.

    GNU make keeps the last recipe declared for a repeated target and only
    warns. The contract reads the first declaration, so a duplicate could
    replace the gated recipe with a command the checks never saw.
    """
    declarations = [
        line
        for line in makefile_text().splitlines()
        if line.startswith(f"{TARGET}:")
    ]
    assert len(declarations) == 1, (
        f"{TARGET} must be declared once; found {len(declarations)} recipes"
    )


@pytest.mark.parametrize("name", REGRESSIONS)
def test_each_regression_is_a_real_test_in_the_scanned_rust_sources(
    name: str,
) -> None:
    """The selected names must still exist as tests in the Rust source.

    This is a second, independent reading: the Makefile could name a test that
    was renamed in the Rust source, and the zero-test flag would only catch
    that at CI time. Reading the declaration here fails at review time. The
    declaration must carry a test attribute, because a plain function of the
    same name is never registered with the harness.
    """
    sources = (
        ROOT / "chutoro-core" / "src" / "hnsw" / "cpu" / "trim_tests"
    ).glob("*.rs")
    declarations = [path.read_text(encoding="utf-8") for path in sources]
    assert any(declares_test(source, name) for source in declarations), (
        f"{name} must still be declared as a test under cpu/trim_tests"
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(
            "#[test]\nfn verify() {}",
            True,
            id="a-top-level-test",
        ),
        pytest.param(
            "mod tests {\n    #[test]\n    fn verify() {}\n}",
            True,
            id="an-indented-test",
        ),
        pytest.param("fn verify() {}", False, id="an-unattributed-function"),
        pytest.param(
            'const S: &str = "#[test]\\nfn verify() {}";',
            False,
            id="a-declaration-in-a-literal",
        ),
        pytest.param(
            "fn helper() {}\nfn verify() {}",
            False,
            id="a-function-below-an-untested-sibling",
        ),
        pytest.param(
            "fn verify() {}\n#[test]\nfn verify() {}",
            True,
            id="a-test-below-an-untested-namesake",
        ),
    ],
)
def test_the_declaration_check_reads_the_attribute_above_the_name(
    source: str, expected: bool
) -> None:
    """A name alone is not a registered test, and indentation changes nothing.

    ``--no-tests=fail`` cannot fire for a function the harness never
    discovered, so an unattributed function must not satisfy the check. A
    declaration nested in a ``mod tests`` block carries indentation before its
    ``fn``, which a walk that started at the keyword would read as the first
    non-attribute line and stop at. An unattributed namesake above a real test
    must not answer for it either: the file does declare the regression.
    """
    assert declares_test(source, "verify") is expected, (
        f"declares_test must return {expected} for {source!r}"
    )