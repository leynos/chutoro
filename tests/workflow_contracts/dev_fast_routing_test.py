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
    """Return each evaluated probe-cargo recipe with its shell tokens."""
    invocations: list[tuple[str, list[str]]] = []
    for line in output.splitlines():
        tokens = shlex.split(line)
        if "probe-cargo" in tokens:
            cargo_index = tokens.index("probe-cargo")
            invocations.append((line, tokens[cargo_index:]))
    return invocations


def _pinned_toolchain() -> str:
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
    output = _make_output(target, overrides=("DEV_FAST_HOST_OS=Linux",))
    _assert_routed_cargo(output, verbs)
    if target == "dev-test":
        _assert_test_rustflags(output, "Linux")


@pytest.mark.parametrize("host_os", ("Linux", "Darwin"))
def test_development_test_route_preserves_warning_and_platform_flags(
    host_os: str,
) -> None:
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
        pytest.param("verus", "scripts/run-verus.sh", id="verus"),
        pytest.param("lint-whitaker", "whitaker", id="whitaker"),
    ),
)
def test_standard_targets_do_not_select_the_fragment(
    target: str, expected_command: str
) -> None:
    """Every merge gate must stay on the repository's stable build standard."""
    output = _make_output(target, overrides=("WHITAKER=whitaker-probe",))
    _assert_unfragmented(output, target)
    assert expected_command in output, (
        f"{target} did not emit its expected command {expected_command!r}: {output!r}"
    )
    if expected_command == "probe-cargo":
        invocations = _cargo_invocations(output)
        assert invocations, f"{target} must use injected CARGO=probe-cargo"
        assert all(tokens[0] == "probe-cargo" for _, tokens in invocations), (
            f"{target} must keep CARGO injectable: {invocations!r}"
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
