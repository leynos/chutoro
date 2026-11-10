"""Contract tests for the workspace doctest gate.

`cargo nextest` does not run doctests, so a test gate built only on nextest never
compiles the examples in a crate's documentation. The Makefile therefore has to
expose a `test-doc` target that runs `cargo test --doc` over the workspace with
all features and warnings denied, `make test` has to run the same check after
nextest, and the CI build-test job has to invoke it before the nextest-based
coverage step. The final test pins the CPU HNSW sources directly: an ignored
doctest is not compiled at all, which is the gap this gate exists to close.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import re
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