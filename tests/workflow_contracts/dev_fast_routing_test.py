"""Contract tests for explicit development Cargo routing."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE_PATH = REPO_ROOT / "Makefile"
DEV_FAST_TOOLCHAIN_PATH = REPO_ROOT / "tools" / "dev-fast" / "TOOLCHAIN"
DEV_FAST_CONFIG = "tools/dev-fast/config.toml"
STABLE_VERIFY_FLAGS = "-D warnings -Dmissing_docs -Dmissing_crate_level_docs"
EXPECTED_NEXTTEST_FILTER = (
    "not kind(bench) & not (package(chutoro-core) & binary(result_api_surface)) & "
    "not (package(chutoro-core) & binary(session_api_surface)) & "
    "not (package(chutoro-providers-dense) & "
    "test(portable_simd_without_feature_is_rejected))"
)


def _make_output(
    target: str,
    *,
    overrides: tuple[str, ...] = (),
    makefile: Path = MAKEFILE_PATH,
    jobs: int | None = None,
) -> str:
    """Return evaluated Make recipes for one target without executing them."""
    command = ["make", "--no-print-directory", "--dry-run"]
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


def _assert_routed_cargo(output: str, expected_verbs: tuple[str, ...]) -> None:
    """Require every emitted Cargo command to select the pinned fragment."""
    toolchain = DEV_FAST_TOOLCHAIN_PATH.read_text(encoding="utf-8").strip()
    assert toolchain, f"{DEV_FAST_TOOLCHAIN_PATH} must contain a toolchain pin"
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
            f"every debug Cargo invocation must begin with "
            f"'probe-cargo {' '.join(expected_prefix)}'; got {line!r}"
        )


def _assert_nextest_selects_fragment(output: str) -> None:
    """Require nextest's Cargo subprocess to select the development fragment."""
    invocations = [
        (line, tokens)
        for line, tokens in _cargo_invocations(output)
        if "nextest" in tokens
    ]
    assert len(invocations) == 1, (
        f"expected one nextest Cargo invocation, found {invocations!r}"
    )
    line, tokens = invocations[0]
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
    """Check development nextest retains warning policy and host linker flags."""
    invocations = _cargo_invocations(output)
    assert invocations, "the Make recipe must emit an injected Cargo command"
    _assert_nextest_selects_fragment(output)
    line, tokens = next(
        (line, tokens)
        for line, tokens in invocations
        if "nextest" in tokens
    )
    toolchain = DEV_FAST_TOOLCHAIN_PATH.read_text(encoding="utf-8").strip()
    assert tokens[1:5] == [
        f"+{toolchain}",
        "--config",
        DEV_FAST_CONFIG,
        "nextest",
    ], f"test must use the selected route: {line!r}"
    assignments = [
        token for token in shlex.split(line) if token.startswith("RUSTFLAGS=")
    ]
    assert len(assignments) == 1, f"test must set RUSTFLAGS once: {line!r}"
    rustflags = assignments[0].partition("=")[2]
    required_flags = ("-D warnings", "-Dmissing_docs", "-Dmissing_crate_level_docs")
    missing = [flag for flag in required_flags if flag not in rustflags]
    assert not missing, f"test RUSTFLAGS lost {missing!r}: {line!r}"
    has_mold = "-Clink-arg=-fuse-ld=mold" in rustflags
    assert has_mold is (host_os == "Linux"), (
        f"test RUSTFLAGS must select mold only on Linux; "
        f"DEV_FAST_HOST_OS={host_os!r}, got {rustflags!r}"
    )


def _assert_test_boundary(output: str, host_os: str) -> None:
    """Check accelerated nextest and the stable byte-exact verification runs."""
    invocations = _cargo_invocations(output)
    assert len(invocations) == 3, (
        f"expected three test Cargo invocations in order, got {invocations!r}"
    )
    assert all(tokens[0] == "probe-cargo" for _, tokens in invocations), (
        f"CARGO=probe-cargo must reach every test command: {invocations!r}"
    )

    nextest_line, nextest_tokens = invocations[0]
    toolchain = DEV_FAST_TOOLCHAIN_PATH.read_text(encoding="utf-8").strip()
    assert nextest_tokens[1:5] == [
        f"+{toolchain}",
        "--config",
        DEV_FAST_CONFIG,
        "nextest",
    ], f"first test command must be accelerated nextest: {nextest_line!r}"
    assert "-E" in nextest_tokens, f"nextest must receive its test filter: {nextest_line!r}"
    filter_index = nextest_tokens.index("-E")
    assert nextest_tokens[filter_index + 1] == EXPECTED_NEXTTEST_FILTER, (
        "nextest must exclude exactly the stable API and SIMD diagnostic tests "
        f"while retaining the remaining suite: {nextest_line!r}"
    )

    stable_commands = invocations[1:]
    expected_stable = (
        (
            "test",
            "-p",
            "chutoro-core",
            "--all-features",
            "--test",
            "result_api_surface",
            "--test",
            "session_api_surface",
        ),
        (
            "test",
            "-p",
            "chutoro-providers-dense",
            "--all-features",
            "--test",
            "portable_simd_gating",
            "--",
            "--exact",
            "portable_simd_without_feature_is_rejected",
        ),
    )
    for (line, tokens), expected in zip(stable_commands, expected_stable, strict=True):
        assert "env -u RUSTUP_TOOLCHAIN" in line, (
            f"stable diagnostic verification must clear the repository toolchain: {line!r}"
        )
        assert DEV_FAST_CONFIG not in tokens and f"+{toolchain}" not in tokens, (
            f"stable verification must not select the dev-fast fragment: {line!r}"
        )
        assert tokens[1 : 1 + len(expected)] == list(expected), (
            f"stable diagnostic verification targets changed: {line!r}"
        )
        assignments = [
            token for token in shlex.split(line) if token.startswith("RUSTFLAGS=")
        ]
        assert len(assignments) == 1, f"stable test must set RUSTFLAGS once: {line!r}"
        assert assignments[0] == f"RUSTFLAGS={STABLE_VERIFY_FLAGS}", (
            f"stable verification must keep its exact warning flags: {line!r}"
        )
        assert "mold" not in assignments[0], (
            f"stable verification must not inherit dev-fast linker flags: {line!r}"
        )

    _assert_test_rustflags(output, host_os)


def _assert_unfragmented(output: str, target: str) -> None:
    assert DEV_FAST_CONFIG not in output, (
        f"{target} must not select {DEV_FAST_CONFIG}; got:\n{output}"
    )


@pytest.mark.parametrize(
    ("target", "verbs"),
    (
        pytest.param("build", ("build",), id="standard-debug-build"),
        pytest.param("test", ("nextest",), id="standard-debug-test"),
        pytest.param("typecheck", ("check",), id="standard-debug-typecheck"),
        pytest.param("lint-clippy", ("doc", "clippy"), id="lint-doc-and-clippy"),
        pytest.param("dev-build", ("build",), id="explicit-dev-build"),
        pytest.param("dev-test", ("nextest",), id="explicit-dev-test"),
    ),
)
def test_debug_make_targets_select_the_pinned_fragment(
    target: str, verbs: tuple[str, ...]
) -> None:
    if target in {"test", "dev-test"}:
        output = _make_output(target, overrides=("DEV_FAST_HOST_OS=Linux",))
        _assert_test_boundary(output, "Linux")
    else:
        _assert_routed_cargo(_make_output(target), verbs)


@pytest.mark.parametrize(
    ("target", "host_os"),
    (
        pytest.param("test", "Linux", id="standard-test-linux"),
        pytest.param("test", "Darwin", id="standard-test-non-linux"),
        pytest.param("dev-test", "Linux", id="dev-test-linux"),
        pytest.param("dev-test", "Darwin", id="dev-test-non-linux"),
    ),
)
def test_test_routes_preserve_warning_and_platform_flags(
    target: str, host_os: str
) -> None:
    output = _make_output(target, overrides=(f"DEV_FAST_HOST_OS={host_os}",))
    _assert_test_boundary(output, host_os)


@pytest.mark.parametrize(
    ("target", "expected_command"),
    (
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
def test_non_development_targets_do_not_select_the_fragment(
    target: str, expected_command: str
) -> None:
    output = _make_output(target, overrides=("WHITAKER=whitaker-probe",))
    _assert_unfragmented(output, target)
    assert expected_command in output, (
        f"{target} did not emit its expected command {expected_command!r}: {output!r}"
    )
    if expected_command == "probe-cargo":
        invocations = _cargo_invocations(output)
        assert invocations, f"{target} must use injected CARGO=probe-cargo"
        assert all(tokens[0] == "probe-cargo" for _, tokens in invocations)


def _mutated_makefile(tmp_path: Path, old: str, new: str) -> Path:
    """Write a private Makefile copy after one asserted text mutation."""
    source = MAKEFILE_PATH.read_text(encoding="utf-8")
    assert source.count(old) == 1, f"mutation anchor must occur once: {old!r}"
    mutated = tmp_path / "Makefile"
    mutated.write_text(source.replace(old, new), encoding="utf-8")
    return mutated


def test_route_assertion_rejects_a_debug_command_without_the_fragment(
    tmp_path: Path,
) -> None:
    """A missing explicit selection is caught using evaluated Make output."""
    makefile = _mutated_makefile(
        tmp_path,
        "DEV_CARGO = $(CARGO) +$(DEV_FAST_TOOLCHAIN) --config $(DEV_FAST_CONFIG)",
        "DEV_CARGO = $(CARGO) +$(DEV_FAST_TOOLCHAIN)",
    )
    output = _make_output("build", makefile=makefile)
    with pytest.raises(AssertionError, match="must begin with"):
        _assert_routed_cargo(output, ("build",))


def test_unfragmented_assertion_rejects_a_contaminated_bench_command(
    tmp_path: Path,
) -> None:
    """A negative-route regression is caught without editing the live Makefile."""
    makefile = _mutated_makefile(
        tmp_path,
        "$(CARGO) bench -p chutoro-benches",
        "$(CARGO) --config tools/dev-fast/config.toml bench -p chutoro-benches",
    )
    output = _make_output("bench", makefile=makefile)
    with pytest.raises(AssertionError, match="must not select"):
        _assert_unfragmented(output, "bench")


def test_test_flag_assertion_rejects_a_missing_warning_flag(tmp_path: Path) -> None:
    """The effective test environment must retain every required denial."""
    makefile = _mutated_makefile(
        tmp_path,
        "DEV_TEST_RUSTFLAGS = $(strip -D warnings -Dmissing_docs",
        "DEV_TEST_RUSTFLAGS = $(strip -D warnings",
    )
    output = _make_output(
        "test", makefile=makefile, overrides=("DEV_FAST_HOST_OS=Linux",)
    )
    with pytest.raises(AssertionError, match="lost"):
        _assert_test_rustflags(output, "Linux")


@pytest.mark.parametrize(
    "target",
    (
        pytest.param("test", id="standard-test"),
        pytest.param("dev-test", id="explicit-dev-test"),
    ),
)
def test_nextest_assertion_rejects_a_missing_inner_fragment(
    target: str, tmp_path: Path
) -> None:
    """A missing nextest selection fails while outer Cargo routing remains."""
    makefile = _mutated_makefile(
        tmp_path,
        "nextest run --config $(DEV_FAST_CONFIG) --profile",
        "nextest run --profile",
    )
    output = _make_output(
        target,
        makefile=makefile,
        overrides=("DEV_FAST_HOST_OS=Linux",),
    )

    with pytest.raises(
        AssertionError,
        match=r"exactly one tools/dev-fast/config\.toml selection",
    ):
        _assert_test_boundary(output, "Linux")
