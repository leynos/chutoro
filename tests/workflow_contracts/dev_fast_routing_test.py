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

import re
from pathlib import Path

import pytest

from dev_fast_make_support import (
    DEV_FAST_CONFIG,
    MAKEFILE_PATH,
    _assert_routed_cargo,
    _assert_unfragmented,
    _cargo_invocations,
    _make_output,
    _mutated_makefile,
)
from dev_fast_rustflags_support import _assert_test_rustflags


#: The Rust module the Makefile exclusion names. Derived from the imported
#: `MAKEFILE_PATH` rather than recomputing the repository root, so this module
#: keeps a single definition of where the checkout is.
TEST_HELPERS_PATH = (
    MAKEFILE_PATH.parent / "chutoro-core" / "src" / "hnsw" / "cpu" / "test_helpers.rs"
)

POISONED_LOCK_TEST = (
    "hnsw::cpu::test_helpers::tests::graph_helpers_report_poisoning_without_reconfiguring"
)


def _nextest_filter_expression(output: str) -> str:
    """Return the filter expression of the single development nextest run."""
    invocations = [
        (line, tokens)
        for line, tokens in _cargo_invocations(output)
        if "nextest" in tokens
    ]
    assert len(invocations) == 1, (
        f"expected one development nextest invocation, found {invocations!r}"
    )
    _, tokens = invocations[0]
    filter_index = tokens.index("-E")
    return tokens[filter_index + 1]


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
    _assert_development_exclusion(_nextest_filter_expression(output))


def _assert_development_exclusion(expression: str) -> None:
    """Require the development filter to exclude exactly the poisoned-lock case.

    A missing close parenthesis is the failure mode this guards against: the
    expression still *reads* correctly, but nextest rejects it outright with
    `error: expected close parenthesis`, so `make dev-test` runs nothing at
    all. Balance is therefore checked before the textual assertions, which
    would otherwise pass against an unparseable expression.
    """
    depth = 0
    for character in expression:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            assert depth >= 0, f"dev-test filter closes an unopened group: {expression!r}"
    assert depth == 0, (
        f"dev-test filter leaves {depth} unclosed group(s), which nextest "
        f"rejects: {expression!r}"
    )
    assert f"test(={POISONED_LOCK_TEST})" in expression, (
        f"dev-test must exclude the poisoned-lock regression: {expression!r}"
    )
    assert expression.startswith("not kind(bench) & not ("), (
        f"dev-test must filter benchmarks and then one named case: {expression!r}"
    )


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
    """
    case = POISONED_LOCK_TEST.rsplit("::", 1)[-1]
    source = TEST_HELPERS_PATH.read_text(encoding="utf-8")
    assert _declared_in_test_module(source, case), (
        f"POISONED_LOCK_TEST names {case!r}, but {TEST_HELPERS_PATH.name} does "
        f"not declare it inside `mod tests`; update the Makefile exclusion "
        f"with the rename"
    )


def _strip_comments(source: str) -> str:
    """Blank out Rust comments while preserving character offsets.

    Offsets have to survive because the caller compares positions in this text
    against positions it found in the original. Each comment character becomes
    a space, and newlines are kept so line anchors still behave.

    Block comments are handled before line comments so a `//` inside `/* */`
    is not mistaken for a line comment, and `//` before `/*` so a block
    delimiter inside a line comment is not treated as a block.
    """
    without_blocks = re.sub(
        r"/\*.*?\*/",
        lambda match: re.sub(r"[^\n]", " ", match.group(0)),
        source,
        flags=re.DOTALL,
    )
    return re.sub(
        r"//[^\n]*",
        lambda match: " " * len(match.group(0)),
        without_blocks,
    )


def _module_body_span(source: str, name: str) -> tuple[int, int] | None:
    """Return the ``(open, close)`` offsets of ``mod name``'s braces, or None.

    Brace counting is depth-based from the module's opening brace, so a nested
    block, an `impl`, or a nested module inside `mod tests` cannot end the
    span early. A `}` in a string literal would break this, which is why the
    source is comment-stripped first; the file has no such literal today.
    """
    header = re.search(rf"^mod\s+{re.escape(name)}\s*\{{", source, flags=re.MULTILINE)
    if not header:
        return None
    body_start = header.end() - 1
    depth = 0
    for offset in range(body_start, len(source)):
        character = source[offset]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return body_start, offset
    return None


def _declared_in_test_module(source: str, case: str) -> bool:
    """Is ``case`` declared as a function inside ``mod tests``?

    The contract this backs is narrow: nextest selects the path
    ``...::tests::<case>``, so the function must exist in that module. A plain
    substring probe accepts far too much — a commented-out declaration, a
    longer name that merely ends in the case, a same-named helper beside the
    code, or a declaration after `mod tests` has closed. Each leaves the
    Makefile exclusion naming nothing, so `dev-test` keeps running the case
    that aborts the Cranelift process.
    """
    stripped = _strip_comments(source)
    span = _module_body_span(stripped, "tests")
    if span is None:
        return False
    body_start, body_close = span
    pattern = rf"^\s*(?:pub\s+)?(?:async\s+)?fn\s+{re.escape(case)}\s*\("
    return any(
        body_start < match.start() < body_close
        for match in re.finditer(pattern, stripped, flags=re.MULTILINE)
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
        _assert_development_exclusion(expression)


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
        _assert_development_exclusion(_nextest_filter_expression(output))


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


_POISONED_CASE = POISONED_LOCK_TEST.rsplit("::", 1)[-1]


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        pytest.param(
            f"#[cfg(test)]\nmod tests {{\n    fn {_POISONED_CASE}() {{}}\n}}\n",
            True,
            id="declared-inside-mod-tests",
        ),
        pytest.param(
            f"mod tests {{\n    fn other() {{}}\n}}\n\nfn {_POISONED_CASE}() {{}}\n",
            False,
            id="declared-after-mod-tests-closes",
        ),
        pytest.param(
            f"mod tests {{\n    /*\n    fn {_POISONED_CASE}() {{}}\n    */\n}}\n",
            False,
            id="inside-block-comment",
        ),
        pytest.param(
            f"mod tests {{\n    // fn {_POISONED_CASE}() {{}}\n}}\n",
            False,
            id="inside-line-comment",
        ),
        pytest.param(
            f"mod tests {{\n    fn not_{_POISONED_CASE}() {{}}\n}}\n",
            False,
            id="longer-name-suffix",
        ),
        pytest.param(
            f"fn {_POISONED_CASE}() {{}}\nmod tests {{\n    fn other() {{}}\n}}\n",
            False,
            id="declared-before-mod-tests",
        ),
        pytest.param(
            f"mod tests {{\n    mod nested {{\n        fn helper() {{}}\n    }}\n"
            f"    fn {_POISONED_CASE}() {{}}\n}}\n",
            True,
            id="still-inside-after-nested-block",
        ),
        pytest.param(
            f"mod other {{\n    fn {_POISONED_CASE}() {{}}\n}}\n",
            False,
            id="different-module",
        ),
    ),
)
def test_poisoned_lock_declaration_must_be_inside_mod_tests(
    source: str, expected: bool
) -> None:
    """The module check must reject every layout that is not a real test.

    The false rows are the point. Each one is a source the previous `fn
    {case}(` substring probe accepted, and each leaves the Makefile exclusion
    naming nothing while `dev-test` keeps running the Cranelift-aborting case.

    The `still-inside-after-nested-block` row is a control: a nested block must
    not end the module span, so a declaration after it is still inside
    `mod tests` and the probe must keep accepting real code.
    """
    actual = _declared_in_test_module(source, _POISONED_CASE)
    assert actual is expected, (
        f"expected {expected} for this layout, got {actual}:\n{source}"
    )
