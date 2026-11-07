"""Contract tests for the opt-in development Cargo routing.

The Cranelift fragment is a development-speed convenience, never a merge gate.
These contracts pin both halves of that boundary: the two explicit development
targets select the fragment, and every standard gate stays on the repository's
stable pin without it.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE_PATH = REPO_ROOT / "Makefile"
TEST_HELPERS_PATH = REPO_ROOT / "chutoro-core" / "src" / "hnsw" / "cpu" / "test_helpers.rs"
DEV_FAST_TOOLCHAIN_PATH = REPO_ROOT / "tools" / "dev-fast" / "TOOLCHAIN"
DEV_FAST_CONFIG = "tools/dev-fast/config.toml"
REQUIRED_TEST_DENIALS = ("-D warnings", "-Dmissing_docs", "-Dmissing_crate_level_docs")


def _make_output(
    target: str,
    *,
    overrides: tuple[str, ...] = (),
    makefile: Path = MAKEFILE_PATH,
    jobs: int | None = None,
) -> str:
    """Return evaluated Make recipes for one target without executing them.

    ``--always-make`` is what makes the reading independent of the checkout:
    without it a target whose file already exists prints nothing and the
    contract would silently pass over an empty script.
    """
    command = ["make", "--no-print-directory", "--dry-run", "--always-make"]
    if jobs is not None:
        command.append(f"-j{jobs}")
    command.extend(("--file", str(makefile), *overrides, target, "CARGO=probe-cargo"))
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"make dry-run for {target!r} failed with {completed.returncode}: "
        f"{completed.stderr}"
    )
    return completed.stdout


def _cargo_invocations(output: str) -> list[tuple[str, list[str]]]:
    """Return each evaluated probe-cargo recipe with its shell tokens.

    A recipe may chain several Cargo commands on one line - `make test` runs
    `$(CARGO) fetch --locked && ... $(CARGO) nextest run ...` - so every
    occurrence starts a new invocation and ends where the next one begins.
    Reading the line as a single command would report `fetch` as the verb and
    hide the nextest run behind it.
    """
    invocations: list[tuple[str, list[str]]] = []
    for line in output.splitlines():
        tokens = shlex.split(line)
        starts = [index for index, token in enumerate(tokens) if token == "probe-cargo"]
        for position, start in enumerate(starts):
            end = starts[position + 1] if position + 1 < len(starts) else len(tokens)
            invocations.append((line, tokens[start:end]))
    return invocations


def _pinned_toolchain() -> str:
    """Return the toolchain channel pinned for the development route."""
    toolchain = DEV_FAST_TOOLCHAIN_PATH.read_text(encoding="utf-8").strip()
    assert toolchain, f"{DEV_FAST_TOOLCHAIN_PATH} must contain a toolchain pin"
    return toolchain


def _assert_routed_cargo(output: str, expected_verbs: tuple[str, ...]) -> None:
    """Require every emitted Cargo command to select the pinned fragment."""
    toolchain = _pinned_toolchain()
    assert (REPO_ROOT / DEV_FAST_CONFIG).is_file(), (
        f"the selected Cargo fragment is missing: {DEV_FAST_CONFIG}"
    )

    invocations = _cargo_invocations(output)
    assert len(invocations) == len(expected_verbs), (
        f"expected Cargo verbs {expected_verbs!r}, found "
        f"{[tokens for _, tokens in invocations]!r}"
    )
    for (line, tokens), verb in zip(invocations, expected_verbs, strict=True):
        expected_prefix = [f"+{toolchain}", "--config", DEV_FAST_CONFIG, verb]
        assert tokens[1 : 1 + len(expected_prefix)] == expected_prefix, (
            f"every development Cargo invocation must begin with "
            f"'probe-cargo {' '.join(expected_prefix)}'; got {line!r}"
        )


def _assert_unfragmented(output: str, target: str) -> None:
    """Require a standard target to stay off the development fragment."""
    assert DEV_FAST_CONFIG not in output, (
        f"{target} must not select {DEV_FAST_CONFIG}; got:\n{output}"
    )


def _assert_nextest_selects_fragment(line: str, tokens: list[str]) -> None:
    """Require nextest's inner Cargo invocation to select the fragment."""
    nextest_args = tokens[tokens.index("nextest") + 1 :]
    profile_index = nextest_args.index("--profile")
    config_indices = [
        index for index, token in enumerate(nextest_args) if token == "--config"
    ]
    assert (
        len(config_indices) == 1
        and config_indices[0] < profile_index
        and nextest_args[config_indices[0] + 1] == DEV_FAST_CONFIG
    ), (
        f"nextest must pass exactly one {DEV_FAST_CONFIG} selection "
        f"before --profile: {line!r}"
    )


def _assert_nextest_scope(line: str, tokens: list[str]) -> None:
    """Check the development run covers the same suite as the gate.

    ``dev-test`` is a convenience, not evidence for merge, but it is worthless
    if it silently narrows the suite. Dropping ``--all-targets`` or
    ``--all-features`` would skip tests, ``--all-targets`` alone is not enough
    without ``run``, and omitting the benchmark filter makes nextest attempt
    cases the standard gate excludes.

    The filter is not pinned to a literal here: the development route also
    excludes the Cranelift-incompatible poisoned-lock regression, so the
    precise expression is owned by `_assert_development_exclusion`. This check
    only requires the benchmark exclusion to be present.
    """
    nextest_args = tokens[tokens.index("nextest") + 1 :]
    assert nextest_args[0] == "run", (
        f"dev-test must invoke 'nextest run', got {nextest_args[0]!r}: {line!r}"
    )
    for required in ("--all-targets", "--all-features"):
        assert required in nextest_args, (
            f"dev-test must pass {required} so the suite matches the gate: {line!r}"
        )
    assert "-E" in nextest_args, f"dev-test must filter benchmarks: {line!r}"
    filter_index = nextest_args.index("-E")
    assert nextest_args[filter_index + 1].startswith("not kind(bench)"), (
        f"dev-test must exclude benchmarks: {line!r}"
    )


def _assert_test_rustflags(output: str, host_os: str) -> None:
    """Check the development nextest run keeps the warning and linker policy."""
    invocations = [
        (line, tokens) for line, tokens in _cargo_invocations(output) if "nextest" in tokens
    ]
    assert len(invocations) == 1, (
        f"expected one development nextest invocation, found {invocations!r}"
    )
    line, tokens = invocations[0]
    toolchain = _pinned_toolchain()
    assert tokens[1:5] == [
        f"+{toolchain}",
        "--config",
        DEV_FAST_CONFIG,
        "nextest",
    ], f"dev-test must use the selected route: {line!r}"
    _assert_nextest_selects_fragment(line, tokens)
    _assert_nextest_scope(line, tokens)
    assignments = [
        token for token in shlex.split(line) if token.startswith("RUSTFLAGS=")
    ]
    assert len(assignments) == 1, f"dev-test must set RUSTFLAGS once: {line!r}"
    rustflags = assignments[0].partition("=")[2]
    missing = [flag for flag in REQUIRED_TEST_DENIALS if flag not in rustflags]
    assert not missing, f"dev-test RUSTFLAGS lost {missing!r}: {line!r}"
    has_mold = "-Clink-arg=-fuse-ld=mold" in rustflags
    assert has_mold is (host_os == "Linux"), (
        f"dev-test RUSTFLAGS must select mold only on Linux; "
        f"DEV_FAST_HOST_OS={host_os!r}, got {rustflags!r}"
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
    assert f"fn {case}(" in TEST_HELPERS_PATH.read_text(encoding="utf-8"), (
        f"POISONED_LOCK_TEST names {case!r}, but no such test exists in "
        f"{TEST_HELPERS_PATH.name}; update the Makefile exclusion with the rename"
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
    if target == "dev-test":
        _assert_test_rustflags(output, "Linux")


@pytest.mark.parametrize("host_os", ("Linux", "Darwin"))
def test_development_test_route_preserves_warning_and_platform_flags(
    host_os: str,
) -> None:
    """`dev-test` must restate the denials, and the Linux linker, itself.

    An assigned `RUSTFLAGS` replaces the fragment's `rustflags` table, so the
    route would silently lose both if it did not repeat them.
    """
    output = _make_output("dev-test", overrides=(f"DEV_FAST_HOST_OS={host_os}",))
    _assert_test_rustflags(output, host_os)


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


def _mutated_makefile(tmp_path: Path, old: str, new: str) -> Path:
    """Write a private Makefile copy after one asserted text mutation."""
    source = MAKEFILE_PATH.read_text(encoding="utf-8")
    assert source.count(old) == 1, f"mutation anchor must occur once: {old!r}"
    mutated = tmp_path / "Makefile"
    mutated.write_text(source.replace(old, new), encoding="utf-8")
    return mutated


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


def test_test_flag_assertion_rejects_a_missing_denial(tmp_path: Path) -> None:
    """The effective development environment must retain every denial."""
    makefile = _mutated_makefile(
        tmp_path,
        "DEV_TEST_RUSTFLAGS = $(strip -D warnings -Dmissing_docs",
        "DEV_TEST_RUSTFLAGS = $(strip -D warnings",
    )
    output = _make_output(
        "dev-test", makefile=makefile, overrides=("DEV_FAST_HOST_OS=Linux",)
    )
    with pytest.raises(AssertionError, match="lost"):
        _assert_test_rustflags(output, "Linux")


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
