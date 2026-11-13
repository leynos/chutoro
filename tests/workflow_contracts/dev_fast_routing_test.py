"""Contract tests for the opt-in development Cargo routing.

The Cranelift fragment is a development-speed convenience, never a merge gate.
These contracts pin both halves of that boundary: the two explicit development
targets select the fragment, and every standard gate stays on the repository's
stable pin without it.

The evaluated-output helpers live in ``dev_fast_make_support``; this module
owns only the routing decisions built on top of them.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dev_fast_filter_contract import (
    POISONED_LOCK_TEST,
    assert_development_exclusion,
    wider_development_filters,
)
from dev_fast_make_support import (
    DEV_FAST_CONFIG,
    MAKEFILE_PATH,
    _assert_routed_cargo,
    _assert_unfragmented,
    _cargo_invocations,
    _make_output,
    _mutated_makefile,
    _nextest_filter_expression,
)
from dev_fast_rustflags_support import _assert_test_rustflags
from rust_declaration_reading import declared_in_module


#: The Rust module the Makefile exclusion names. Derived from the imported
#: `MAKEFILE_PATH` rather than recomputing the repository root, so this module
#: keeps a single definition of where the checkout is.
TEST_HELPERS_PATH = (
    MAKEFILE_PATH.parent / "chutoro-core" / "src" / "hnsw" / "cpu" / "test_helpers.rs"
)

#: Re-exported from `dev_fast_filter_contract`, which builds the expected
#: filter from the same name, so a rename cannot leave the two disagreeing.
__all__ = ["POISONED_LOCK_TEST"]


def test_development_test_excludes_only_the_poisoned_lock_regression() -> None:
    """The Cranelift route gives up the one case its backend cannot run.

    Cranelift aborts on a panic it cannot unwind
    (`fatal runtime error: failed to initiate panic, error 5`), and the
    regression in `hnsw::cpu::test_helpers::tests` poisons a lock by panicking
    under `catch_unwind`. Excluding it keeps `dev-test` usable. The exclusion
    is written once, through `POISONED_LOCK_TEST`, so a renamed test is a
    one-line Makefile edit and the stable gate can keep running it.
    """
    output = _make_output("dev-test", overrides=("DEV_FAST_HOST_OS=Linux",))
    assert_development_exclusion(_nextest_filter_expression(output))


@pytest.mark.parametrize(("case", "expression"), wider_development_filters())
def test_filter_contract_rejects_a_wider_development_exclusion(
    case: str, expression: str
) -> None:
    """Adding an exclusion must fail the contract, not slip past it.

    The measured filter selects fewer tests than intended, which is the
    failure a substring check cannot see: the poisoned-lock clause is still
    present and the expected prefix still matches, so only the exact
    comparison in `assert_development_exclusion` rejects it. Without that
    comparison the suite enforces "a case is named", never the "and nothing
    else" its docstring promises.
    """
    with pytest.raises(AssertionError):
        assert_development_exclusion(expression)


def _assert_poisoned_lock_case_is_named_once() -> None:
    """Require the exclusion to be declared in the Makefile, not duplicated."""
    makefile = MAKEFILE_PATH.read_text(encoding="utf-8")
    assert makefile.count(POISONED_LOCK_TEST) == 1, (
        "the poisoned-lock case must be named once, through POISONED_LOCK_TEST"
    )
    assert makefile.count("POISONED_LOCK_TEST") >= 2, (
        "the poisoned-lock case must be referenced by the filter definition"
    )


def test_poisoned_lock_exclusion_names_a_rust_test_that_exists() -> None:
    """The Makefile path must resolve to a real test, not a stale rename.

    `POISONED_LOCK_TEST` is written as a module path with no Rust compiler in
    the loop, so a rename in `test_helpers.rs` would leave the exclusion
    pointing at nothing. nextest treats an unmatched `test(=...)` clause as an
    empty match rather than an error, so `dev-test` would silently keep running
    the regression under Cranelift and abort.

    The reading asks for the declaration at the module's own level, not merely
    somewhere inside it. A case moved into a nested module has a longer path
    than the one the filter names, and nextest selects by the full trailing
    segment, so that move is a missing exclusion too.

    Both the module and the case come from the path itself. Hard-coding
    `"tests"` here would keep passing against the old module name after a
    rename, so the contract would go on vouching for a path the Makefile no
    longer names.
    """
    *_, module, case = POISONED_LOCK_TEST.split("::")
    source = TEST_HELPERS_PATH.read_text(encoding="utf-8")
    assert declared_in_module(source, module, case), (
        f"POISONED_LOCK_TEST names {case!r}, but {TEST_HELPERS_PATH.name} does "
        f"not declare it directly inside `mod {module}`; update the Makefile "
        f"exclusion with the rename"
    )


@pytest.mark.parametrize(
    ("target", "verbs"),
    (
        pytest.param("dev-build", ("build",), id="explicit-dev-build"),
        pytest.param("dev-test", ("nextest",), id="explicit-dev-test"),
    ),
)
def test_development_targets_select_the_pinned_fragment(
    target: str, verbs: tuple[str, ...]
) -> None:
    """Each development target must reach Cargo through the pinned fragment.

    The fragment is the only thing that makes these targets a speed route, so
    the routed invocation has to carry both the toolchain pin and the config.
    """
    output = _make_output(target, overrides=("DEV_FAST_HOST_OS=Linux",))
    _assert_routed_cargo(output, verbs)


@pytest.mark.parametrize(
    ("target", "expected_command"),
    (
        pytest.param("build", "probe-cargo", id="standard-debug-build"),
        pytest.param("test", "probe-cargo", id="standard-test"),
        pytest.param("typecheck", "probe-cargo", id="standard-typecheck"),
        pytest.param("lint-clippy", "probe-cargo", id="standard-lint"),
        pytest.param("release", "probe-cargo", id="release-build"),
        pytest.param("fmt", "probe-cargo", id="format"),
        pytest.param("check-fmt", "probe-cargo", id="format-check"),
        pytest.param("bench", "probe-cargo", id="bench"),
        pytest.param("kani", "probe-cargo", id="kani"),
        pytest.param("kani-full", "probe-cargo", id="full-kani"),
        # The verifier and Dylint recipes delegate to trunk-owned tooling. What
        # this contract owns is that they never select the development
        # fragment, so it asserts the recipe emitted a command without pinning
        # a helper whose path and arguments change under the stack.
        pytest.param("verus", None, id="verus"),
        pytest.param("lint-whitaker", None, id="whitaker"),
    ),
)
def test_standard_targets_do_not_select_the_fragment(
    target: str, expected_command: str | None
) -> None:
    """Every merge gate must stay on the repository's stable build standard."""
    output = _make_output(target, overrides=("WHITAKER=whitaker-probe",))
    _assert_unfragmented(output, target)
    assert output.strip(), f"{target} must emit a command: {output!r}"
    if expected_command is None:
        return
    assert expected_command in output, (
        f"{target} did not emit its expected command {expected_command!r}: {output!r}"
    )
    invocations = _cargo_invocations(output)
    assert invocations, f"{target} must use injected CARGO=probe-cargo"
    assert all(tokens[0] == "probe-cargo" for _, tokens in invocations), (
        f"{target} must keep CARGO injectable: {invocations!r}"
    )


def test_development_targets_declare_phony_beside_their_recipes() -> None:
    """Keep the opt-in declarations out of the file-wide `.PHONY` line.

    The file-wide list is one long line that concurrent work also edits. When
    this feature appended to it, a branch that added the development targets
    conflicted with an unrelated branch that added the prover targets to the
    same line. GitHub builds a stacked pull request's merge ref from the
    accumulated trunk, so that conflict stopped the merge ref being created and
    no pull-request workflow ran at all. Declaring the three targets beside
    their recipes keeps this feature and its declarations in one hunk.
    """
    makefile_lines = MAKEFILE_PATH.read_text(encoding="utf-8").splitlines()
    phony_lines = [line for line in makefile_lines if line.startswith(".PHONY:")]

    development_targets = {"dev-build", "dev-test", "install-dev-fast"}
    declared: list[str] = []
    for line in phony_lines:
        declared.extend(token for token in line[len(".PHONY:") :].split())

    assert development_targets <= set(declared), (
        f"development targets must be phony: {sorted(development_targets - set(declared))}"
    )
    file_wide = phony_lines[0]
    leaked = development_targets & set(file_wide[len(".PHONY:") :].split())
    assert not leaked, (
        "development targets must declare themselves beside their recipes, "
        f"not in the file-wide line: {sorted(leaked)}"
    )


def test_route_assertion_rejects_a_development_command_without_the_fragment(
    tmp_path: Path,
) -> None:
    """A missing explicit selection is caught using evaluated Make output."""
    makefile = _mutated_makefile(
        tmp_path,
        "DEV_CARGO = $(CARGO) +$(DEV_FAST_TOOLCHAIN) --config $(DEV_FAST_CONFIG)",
        "DEV_CARGO = $(CARGO) +$(DEV_FAST_TOOLCHAIN)",
    )
    output = _make_output("dev-build", makefile=makefile)
    with pytest.raises(AssertionError, match="must begin with"):
        _assert_routed_cargo(output, ("build",))


def test_poisoned_lock_exclusion_is_declared_once() -> None:
    """A duplicated test path drifts; the Makefile names the case once."""
    _assert_poisoned_lock_case_is_named_once()


def test_test_flag_assertion_rejects_a_missing_development_exclusion(
    tmp_path: Path,
) -> None:
    """Dropping the exclusion reintroduces the Cranelift abort onto dev-test."""
    makefile = _mutated_makefile(
        tmp_path,
        "DEV_TEST_FILTER = not kind(bench) & not "
        "(package(chutoro-core) & test(=$(POISONED_LOCK_TEST)))",
        "DEV_TEST_FILTER = not kind(bench)",
    )
    output = _make_output(
        "dev-test", makefile=makefile, overrides=("DEV_FAST_HOST_OS=Linux",)
    )
    expression = _nextest_filter_expression(output)
    assert POISONED_LOCK_TEST not in expression, (
        "the mutation must actually drop the exclusion, or the test proves nothing"
    )
    with pytest.raises(AssertionError, match="must exclude the poisoned-lock"):
        assert_development_exclusion(expression)


@pytest.mark.parametrize(
    "extra",
    (
        pytest.param("-E 'all()'", id="short-form-separated"),
        pytest.param("-Eall()", id="short-form-attached"),
        pytest.param("--filterset 'all()'", id="long-form-separated"),
        pytest.param("--filterset=all()", id="long-form-attached"),
        # The pinned 0.9.143 declares `--filter-expr` as a clap alias of
        # `--filterset`, so it appends to the same list rather than being a
        # separate feature. Nextest's own help text omits the alias, which is
        # exactly why a reader built from `--help` output would miss it.
        pytest.param("--filter-expr 'all()'", id="alias-separated"),
        pytest.param("--filter-expr=all()", id="alias-attached"),
    ),
)
def test_additional_nextest_filter_is_rejected(tmp_path: Path, extra: str) -> None:
    """A second filter argument must be reported, not silently accepted.

    Nextest does not treat a repeated filter as an override: it unions the
    expressions, so a recipe that appended `-E 'all()'` would run every case
    the development filter excludes, including the poisoned-lock regression
    that aborts under Cranelift. A parser reading only the first `-E` would
    keep reporting the intended exclusion while the real run reselected it.

    Every accepted spelling is a row: the four textual forms of the documented
    option and the two of its undocumented alias.
    """
    makefile = _mutated_makefile(
        tmp_path,
        "-E '$(DEV_TEST_FILTER)'",
        f"-E '$(DEV_TEST_FILTER)' {extra}",
    )
    output = _make_output(
        "dev-test", makefile=makefile, overrides=("DEV_FAST_HOST_OS=Linux",)
    )
    with pytest.raises(AssertionError, match="exactly one nextest filter"):
        _nextest_filter_expression(output)


def test_exclusion_assertion_rejects_an_unclosed_filter_group(
    tmp_path: Path,
) -> None:
    """An unbalanced filter is reported, not passed over as a textual match.

    Regression: the development filter previously omitted its final close
    parenthesis. The textual assertions still matched, so the contract suite
    stayed green while every `make dev-test` invocation failed to parse.
    """
    makefile = _mutated_makefile(
        tmp_path,
        "DEV_TEST_FILTER = not kind(bench) & not "
        "(package(chutoro-core) & test(=$(POISONED_LOCK_TEST)))",
        "DEV_TEST_FILTER = not kind(bench) & not "
        "(package(chutoro-core) & test(=$(POISONED_LOCK_TEST))",
    )
    output = _make_output(
        "dev-test", makefile=makefile, overrides=("DEV_FAST_HOST_OS=Linux",)
    )
    with pytest.raises(AssertionError, match="unclosed group"):
        assert_development_exclusion(_nextest_filter_expression(output))


def test_exclusion_assertion_rejects_an_inverted_filter_group(
    tmp_path: Path,
) -> None:
    """A filter that isolates the excluded case is rejected, not accepted.

    The inverted group keeps every string the positive filter contains, so the
    textual assertions alone cannot tell the two apart: presence checks pass
    and only the polarity differs. What `make dev-test` would then run is the
    poisoned-lock regression and nothing else — the case the Cranelift route
    cannot survive — so this mutation has to fail loudly.
    """
    makefile = _mutated_makefile(
        tmp_path,
        "DEV_TEST_FILTER = not kind(bench) & not "
        "(package(chutoro-core) & test(=$(POISONED_LOCK_TEST)))",
        "DEV_TEST_FILTER = not kind(bench) & not "
        "(package(chutoro-core) & not test(=$(POISONED_LOCK_TEST)))",
    )
    output = _make_output(
        "dev-test", makefile=makefile, overrides=("DEV_FAST_HOST_OS=Linux",)
    )
    expression = _nextest_filter_expression(output)
    assert POISONED_LOCK_TEST in expression, (
        "the mutation must keep the case name, or the polarity is not what fails"
    )
    with pytest.raises(AssertionError, match="not select it alone"):
        assert_development_exclusion(expression)


def test_standard_route_assertion_rejects_a_contaminated_bench_command(
    tmp_path: Path,
) -> None:
    """A negative-route regression is caught without editing the live Makefile."""
    makefile = _mutated_makefile(
        tmp_path,
        "$(CARGO) bench -p chutoro-benches",
        "$(CARGO) --config tools/dev-fast/config.toml bench -p chutoro-benches",
    )
    output = _make_output("bench", makefile=makefile)
    assert DEV_FAST_CONFIG in output, (
        "the mutation must actually select the fragment, or the test proves nothing"
    )
    with pytest.raises(AssertionError, match="must not select"):
        _assert_unfragmented(output, "bench")


def test_nextest_assertion_rejects_a_missing_inner_fragment(tmp_path: Path) -> None:
    """A missing inner nextest selection fails while outer routing remains."""
    makefile = _mutated_makefile(
        tmp_path,
        "nextest run --config $(DEV_FAST_CONFIG) --profile",
        "nextest run --profile",
    )
    output = _make_output(
        "dev-test", makefile=makefile, overrides=("DEV_FAST_HOST_OS=Linux",)
    )

    with pytest.raises(
        AssertionError,
        match=r"exactly one tools/dev-fast/config\.toml selection",
    ):
        _assert_test_rustflags(output, "Linux")
