"""Contract tests for the workspace doctest gate.

`cargo nextest` does not run doctests, so a test gate built only on nextest never
compiles the examples in a crate's documentation. The Makefile therefore has to
expose a `test-doc` target that runs `cargo test --doc` over the workspace with
all features and warnings denied, `make test` has to run the same check after
nextest, and the CI build-test job has to invoke it before the nextest-based
coverage step. Textual checks pin that wiring, and end-to-end cases run the
real targets against a scratch workspace with a passing and a failing example,
including through `make test` and the command the CI step declares, so an
advisory gate cannot pass this contract. The final test pins the CPU HNSW
sources directly: an ignored doctest is not compiled at all, which is the gap
this gate exists to close.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = ROOT / "Makefile"
CI = ROOT / ".github" / "workflows" / "ci.yml"
CPU_HNSW_DIR = ROOT / "chutoro-core" / "src" / "hnsw" / "cpu"


def _makefile_text() -> str:
    """Return the Makefile source."""
    return MAKEFILE.read_text(encoding="utf-8")


def _recipe(target: str) -> str:
    """Return the tab-indented recipe lines belonging to *target*.

    Matching the tab-indented run directly is what keeps a search for a flag
    inside the target's own recipe: a plain text search would also find it in a
    neighbouring or unrelated target.
    """
    pattern = re.compile(
        rf"^{re.escape(target)}:[^\n]*\n((?:\t[^\n]*\n)+)", re.MULTILINE
    )
    match = pattern.search(_makefile_text())
    assert match, f"the Makefile must define a {target} recipe"
    return match.group(1)


def test_test_doc_compiles_the_whole_workspace_with_warnings_denied() -> None:
    """Dropping a flag would narrow the gate without any visible failure."""
    commands = [
        line
        for line in _recipe("test-doc").splitlines()
        if "$(CARGO) test --doc" in line
    ]
    assert commands, "test-doc must run `cargo test --doc`"
    command = commands[0]
    assert "--workspace" in command
    assert "--all-features" in command
    assert re.search(r'RUSTFLAGS="[^"]*-D warnings', command), (
        "test-doc must deny warnings through the RUSTFLAGS assignment"
    )


def test_test_doc_is_declared_phony() -> None:
    """A file named `test-doc` must not make the target look up to date."""
    declared = re.findall(
        r"^\.PHONY:((?:.*\\\n)*.*)$", _makefile_text(), re.MULTILINE
    )
    names = " ".join(declared).replace("\\\n", " ").split()
    assert "test-doc" in names, "test-doc must be a .PHONY target"


#: The `make test` command that runs the suite. The optional prefix is the
#: warning-denying environment assignment the recipe puts before the command.
_NEXTEST_COMMAND = re.compile(
    r'^(?:RUSTFLAGS="[^"]*"\s+)?\$\(CARGO\) nextest run\b'
)


def test_make_test_runs_the_doctests_after_nextest() -> None:
    """A local commit gate that stops at nextest leaves examples uncompiled."""
    commands = [line.strip() for line in _recipe("test").splitlines()]
    nextest_at = next(
        (
            index
            for index, command in enumerate(commands)
            if _NEXTEST_COMMAND.match(command)
        ),
        None,
    )
    doctests_at = next(
        (
            index
            for index, command in enumerate(commands)
            if command == "$(MAKE) test-doc"
        ),
        None,
    )
    assert nextest_at is not None, "`make test` must run nextest"
    assert doctests_at is not None, "`make test` must run the doctest target"
    assert nextest_at < doctests_at, "run the doctests after the longer suite"


def test_ci_runs_the_doctest_gate_before_coverage() -> None:
    """The coverage action runs nextest, so the gate has to be its own step."""
    workflow = yaml.safe_load(CI.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["build-test"]["steps"]
    commands = [str(step.get("run", "")).strip() for step in steps]
    assert "make test-doc" in commands, "CI must run `make test-doc`"
    doctests_at = commands.index("make test-doc")

    # A guarded or non-blocking step still appears in the command list, but the
    # build never fails on it, so the gate would be advisory rather than
    # required.
    doctest_step = steps[doctests_at]
    assert "if" not in doctest_step, "a guarded doctest step does not gate CI"
    assert "continue-on-error" not in doctest_step, (
        "a non-blocking doctest step does not gate CI"
    )

    setup_rust = next(
        (
            i
            for i, step in enumerate(steps)
            if "setup-rust" in str(step.get("uses", ""))
        ),
        None,
    )
    assert setup_rust is not None, "CI must use the setup-rust action"
    assert setup_rust < doctests_at, "the toolchain must exist before the gate"

    coverage_at = next(
        (
            i
            for i, step in enumerate(steps)
            if "generate-coverage" in str(step.get("uses", ""))
        ),
        None,
    )
    assert coverage_at is not None, "CI must use the generate-coverage action"
    assert doctests_at < coverage_at, (
        "the doctest step must run before the nextest-only coverage step, so a "
        "failure is reported against the gate rather than buried in the report"
    )


#: A fence opener and its info string, after any doc-comment marker is removed.
_FENCE_OPENER = re.compile(r"^(?:`{3,}|~{3,})(?P<info>.*)$")


def _strip_doc_marker(line: str) -> str:
    """Return *line* without its leading `///` or `//!` marker."""
    stripped = line.strip()
    for marker in ("///", "//!"):
        if stripped.startswith(marker):
            return stripped[len(marker) :].strip()
    return stripped


def _has_ignore_attribute(info: str) -> bool:
    """Return whether a fence info string carries an `ignore` attribute."""
    attributes = (part.strip() for part in info.split(","))
    return any(
        attribute == "ignore" or attribute.startswith("ignore-")
        for attribute in attributes
    )


def _ignored_fences(source: str) -> list[int]:
    """Return the line numbers of doc fences that skip compilation.

    Rustdoc accepts the attribute as a bare `ignore`, as `rust,ignore`, and
    with whitespace around the comma, so matching one literal spelling would
    let another form reintroduce an unchecked example. Matching the attribute
    list also catches the per-target `ignore-<target>` variants, which skip the
    example on the targets they name.
    """
    ignored: list[int] = []
    for number, line in enumerate(source.splitlines(), start=1):
        match = _FENCE_OPENER.match(_strip_doc_marker(line))
        if match and _has_ignore_attribute(match.group("info")):
            ignored.append(number)
    return ignored


@pytest.mark.parametrize(
    "fence",
    ["```ignore", "```rust,ignore", "```rust, ignore", "```ignore-x86_64"],
)
def test_every_ignored_fence_form_is_detected(fence: str) -> None:
    """Rustdoc accepts each spelling, and each one skips compilation."""
    source = f"/// {fence}\n/// let x = 1;\n/// ```\n"
    assert _ignored_fences(source) == [1]


def test_a_plain_fence_is_not_flagged() -> None:
    """Only the `ignore` attribute makes a fence unchecked."""
    assert _ignored_fences("/// ```\n/// let x = 1;\n/// ```\n") == []


def test_the_cpu_hnsw_sources_hold_no_ignored_doctests() -> None:
    """An ignored fence skips compilation, so the example is never checked."""
    offenders = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in sorted(CPU_HNSW_DIR.rglob("*.rs"))
        for number in _ignored_fences(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"ignored doctests bypass compilation: {offenders}"


#: A one-member workspace the target can compile without touching the repository.
_FIXTURE_MANIFEST = """\
[workspace]
members = ["fixture"]
resolver = "2"
"""

_FIXTURE_PACKAGE = """\
[package]
name = "doctest_fixture"
version = "0.1.0"
edition = "2021"
"""


def _fixture_source(expected: int) -> str:
    """Return a fixture crate whose doctest expects *expected*."""
    return f"""\
//! Fixture crate for the doctest gate.
//!
//! ```
//! assert_eq!(doctest_fixture::answer(), {expected});
//! ```

/// Return the fixture answer.
pub const fn answer() -> u32 {{
    42
}}
"""


def _write_doctest_fixture(root: Path, lib_source: str) -> Path:
    """Write a Cargo workspace holding *lib_source* and return its manifest."""
    (root / "fixture" / "src").mkdir(parents=True)
    (root / "Cargo.toml").write_text(_FIXTURE_MANIFEST, encoding="utf-8")
    (root / "fixture" / "Cargo.toml").write_text(_FIXTURE_PACKAGE, encoding="utf-8")
    (root / "fixture" / "src" / "lib.rs").write_text(lib_source, encoding="utf-8")
    return root / "Cargo.toml"


def _cargo_wrapper(root: Path) -> Path:
    """Write a `cargo` shim that runs the subcommand in the manifest's workspace.

    `--manifest-path` belongs to the subcommand, so it cannot be appended to
    the `CARGO` value the recipe expands before `test`. The shim inserts it
    after the subcommand instead.

    `fetch` and `nextest` are stubbed out rather than delegated: the fixture
    workspace has no dependencies to fetch and no nextest suite, and the
    question here is only whether a failing doctest reaches the caller.
    """
    cargo = shutil.which("cargo")
    assert cargo, "the doctest gate test needs cargo"
    wrapper = root / "cargo-fixture.sh"
    wrapper.write_text(
        "#!/bin/sh\n"
        'manifest=$1\n'
        "shift\n"
        'command=$1\n'
        "shift\n"
        "case \"$command\" in\n"
        "  fetch|nextest) exit 0 ;;\n"
        "esac\n"
        f'exec {shlex.quote(cargo)} "$command" --manifest-path "$manifest" "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper


def _run_make(target: str, manifest: Path) -> subprocess.CompletedProcess[str]:
    """Run *target* against the fixture workspace."""
    make = shutil.which("make")
    assert make, "the doctest gate test needs make"
    wrapper = _cargo_wrapper(manifest.parent)
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            make,
            "-C",
            str(ROOT),
            target,
            f"CARGO={shlex.quote(str(wrapper))} {shlex.quote(str(manifest))}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("target", "expected", "should_pass", "marker"),
    [
        # The standalone target stands in for the CI step's command.
        ("test-doc", 42, True, "test result: ok"),
        ("test-doc", 41, False, "test result: FAILED"),
        # `make test` is the local entry point that runs nextest first.
        ("test", 42, True, "test result: ok"),
        ("test", 41, False, "test result: FAILED"),
    ],
)
def test_every_entry_point_propagates_the_doctest_outcome(
    tmp_path: Path,
    target: str,
    expected: int,
    should_pass: bool,
    marker: str,
) -> None:
    """A broken example must fail each entry point, not only the bare target."""
    result = _run_make(target, _write_doctest_fixture(tmp_path, _fixture_source(expected)))
    if should_pass:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0, result.stdout + result.stderr
    assert marker in result.stdout


def _ci_doctest_command() -> list[str]:
    """Return the argv CI's doctest step runs, as the workflow declares it."""
    workflow = yaml.safe_load(CI.read_text(encoding="utf-8"))
    for step in workflow["jobs"]["build-test"]["steps"]:
        if str(step.get("run", "")).strip() == "make test-doc":
            return shlex.split(str(step["run"]))
    message = "CI must declare the doctest step this test exercises"
    raise AssertionError(message)


def test_the_ci_doctest_entry_point_fails_on_a_failing_doctest(
    tmp_path: Path,
) -> None:
    """The exact command CI runs must fail on a broken example.

    The hosted runner itself is not simulated; this executes the command the
    workflow step declares, against the fixture workspace, so a step that
    could not fail would be caught here rather than in the hosted job.
    """
    argv = _ci_doctest_command()
    assert argv[:1] == ["make"], f"unsupported CI invocation: {argv}"
    result = _run_make(argv[1], _write_doctest_fixture(tmp_path, _fixture_source(41)))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "test result: FAILED" in result.stdout