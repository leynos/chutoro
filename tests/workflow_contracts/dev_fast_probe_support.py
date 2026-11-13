"""Build a minimal crate that the real development recipes can compile.

``dev_fast_end_to_end_test`` runs ``dev-build`` and ``dev-test`` rather than
reading a dry run, so it needs a tree those recipes can finish quickly that
still carries the repository's own Make inputs. Building that here keeps the
contract module to its assertions and lets the default-prefix linker contract
reuse the same probe.

The probe copies the real ``Makefile`` and pins rather than a fixture shaped
like them. A recipe and the tree it ships with that drifted apart would then
fail here, which is the point: a fixture that supplies its own pin proves the
recipe works only for inputs no developer has.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import typing as typ
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The probe crate the real recipes build. It is deliberately minimal: the
#: subject is the Make recipe, not the crate, and a tiny tree keeps the
#: compiles inside a test-sized budget. The empty ``[workspace]`` stops the
#: probe joining the checkout's workspace if it is ever written inside it.
#: The crate name is a constant because callers select its compiler lines out
#: of Cargo's verbose output by that name.
PROBE_CRATE = "devprobe"

PROBE_MANIFEST = f"""\
[package]
name = "{PROBE_CRATE}"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "{PROBE_CRATE}"
path = "src/main.rs"

[workspace]
"""

#: What the probe binary prints, so a caller can tell an artifact this route
#: produced from one that merely appeared at the expected path. It matches
#: :data:`PROBE_CRATE` because the source below is written as a plain literal;
#: the assertion in ``assert_probe_inputs_are_real`` keeps the two in step.
PROBE_GREETING = f"{PROBE_CRATE} ok"

PROBE_MAIN = """\
//! Minimal probe crate for the development route.

/// Entry point proving the route builds and links.
fn main() {
    println!("devprobe ok");
}

#[cfg(test)]
mod tests {
    /// Proves the development test route reaches the suite.
    #[test]
    fn the_route_runs_tests() {
        assert_eq!(2 + 2, 4);
    }
}
"""

#: Every file the recipes read at parse time or run. The probe copies each one
#: from the checkout, so a parse-time read added to the Makefile has to be
#: added here as well or the probe fails for a reason the contract explains.
PROBE_INPUTS = (
    "Makefile",
    "tools/dev-fast/TOOLCHAIN",
    "tools/dev-fast/config.toml",
    "tools/kani/VERSION",
)

#: The compiler argument Cargo emits when the fragment's ``codegen-backend``
#: reaches rustc. It appears only in a verbose build, so callers pass
#: ``--verbose`` through ``BUILD_JOBS`` rather than re-deriving the route.
CRANELIFT_ARGUMENT = "codegen-backend=cranelift"

#: The marker Cargo puts on a ``rustc`` invocation for a test target. The
#: development routes build a binary and a test harness from the same crate,
#: and only a line carrying this marker shows that the *test* was compiled
#: with the fragment rather than merely linked from a shared build.
TEST_HARNESS_MARKER = "--test"


def write_probe_repository(directory: Path) -> Path:
    """Write a minimal crate carrying the repository's real Make inputs."""
    (directory / "src").mkdir(parents=True, exist_ok=True)
    (directory / "Cargo.toml").write_text(PROBE_MANIFEST, encoding="utf-8")
    (directory / "src" / "main.rs").write_text(PROBE_MAIN, encoding="utf-8")
    for relative in PROBE_INPUTS:
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO_ROOT / relative).read_bytes())
    return directory


def probe_environment(
    directory: Path, home: Path, *, extra_path: Path | None = None
) -> dict[str, str]:
    """Return an environment that keeps the probe's build state private.

    The probe gets its own target directory and Cargo home, so its compiles
    neither pollute nor queue behind the checkout's cache. The toolchain
    installation is shared through ``RUSTUP_HOME``, because selecting that
    toolchain is exactly what the route under test does; pointing it at the
    private home would make the recipe look for a toolchain that was never
    installed there and fail for the wrong reason.

    ``extra_path`` prepends a directory to ``PATH``, which is how the linker
    contracts place a controlled ``ld.mold`` ahead of the host's own.
    Colour is disabled because nextest writes its run report to standard
    error, and escape sequences would split the summary the callers match on.
    """
    environment = os.environ.copy()
    environment["HOME"] = str(home)
    environment["CARGO_TARGET_DIR"] = str(directory / "target")
    environment["CARGO_HOME"] = str(directory / "cargo-home")
    environment["CARGO_TERM_COLOR"] = "never"
    rustup_home = _rustup_home()
    if rustup_home is not None:
        environment["RUSTUP_HOME"] = str(rustup_home)
    if extra_path is not None:
        environment["PATH"] = f"{extra_path}{os.pathsep}{environment['PATH']}"
    return environment


class RustupHomeProbeError(RuntimeError):
    """``rustup show home`` ran but could not report a usable root."""


def _rustup_home() -> Path | None:
    """Return the shared toolchain root, or ``None`` when rustup is absent.

    The probe redirects ``HOME`` to a private directory, so it also has to
    carry the real toolchain root across: without ``RUSTUP_HOME`` the recipe
    would look for an installation under the private home, find none, and
    fail with an error that points at the toolchain rather than at the probe
    that lost it.

    An absent ``rustup`` is a host that has not provisioned the route, so it
    answers ``None`` and the variable is simply left unset. That is different
    from a ``rustup`` that ran and could not answer: an empty listing or a
    non-zero exit is a broken probe on a host that has the tool, and swallowing
    it would surface later as a misleading missing-toolchain failure. Those
    cases raise :class:`RustupHomeProbeError` naming what went wrong.
    """
    try:
        reported = subprocess.run(
            ["rustup", "show", "home"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if reported.returncode != 0:
        raise RustupHomeProbeError(
            "rustup could not report its home: "
            f"exit {reported.returncode}: {reported.stderr.strip()}"
        )
    if not reported.stdout.strip():
        raise RustupHomeProbeError("rustup show home printed an empty listing")
    return Path(reported.stdout.strip())


#: The make variables every run sets. ``APP`` points the recipe at the probe's
#: binary, and ``BUILD_JOBS`` is where the compiler arguments the backend
#: assertion reads come from. Callers add to this map rather than gaining a
#: parameter each: they are one concept - what this run tells make - and
#: CodeScene's advisory rule counts parameters, not concepts.
BASE_MAKE_VARIABLES: typ.Final[dict[str, str]] = {
    "APP": PROBE_CRATE,
    "BUILD_JOBS": "--verbose",
}


def run_development_target(
    directory: Path,
    environment: dict[str, str],
    target: str,
    make_variables: typ.Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one development recipe against the probe crate in ``directory``.

    ``APP`` points the recipe at the probe's binary and
    ``BUILD_JOBS`` reaches the Cargo invocation with the compiler arguments
    the backend assertion reads. ``CI`` is cleared so the nextest profile is
    the developer's default rather than the gate's.

    ``dev-build`` needs ``--verbose``, but ``dev-test`` needs
    ``--cargo-verbose`` instead: nextest starts its own Cargo, and only the
    latter flag makes that inner build print the line the assertion reads.
    The default suits the direct Cargo route; a nextest caller overrides it
    through ``make_variables``.

    A ``DEV_TEST_FILTER`` in ``make_variables`` replaces the production filter
    for the run. That filter is written against this repository's own sources:
    it scopes the poisoned-lock exclusion with ``package(chutoro-core)``, and
    nextest rejects a ``package(...)`` naming a package the tree does not
    contain (``error: operator didn't match any packages``). The probe crate is
    a single synthetic package, so a caller exercising the recipe's own
    plumbing passes an expression built the same way but without that scope.
    Leaving the default alone is deliberate: changing the production filter to
    suit a fixture would trade a real guarantee for a synthetic one.

    Cargo writes its progress and nextest its run report to standard error,
    leaving standard output for make's echo of the recipe, so callers read the
    stream each assertion belongs to.
    """
    environment = dict(environment)
    environment.pop("CI", None)
    variables = {**BASE_MAKE_VARIABLES, **(make_variables or {})}
    return subprocess.run(
        [
            "make",
            "--no-print-directory",
            "--file",
            str(directory / "Makefile"),
            target,
            *(f"{name}={value}" for name, value in variables.items()),
        ],
        cwd=directory,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def crate_compilations(stderr: str, crate: str) -> list[str]:
    """Return the ``rustc`` command lines Cargo ran for ``crate``.

    Cargo prints each compiler invocation as ``Running `... rustc ...``` on
    standard error when verbose, and nextest prefixes those lines with its own
    run output. Matching the whole stream for a flag would also accept one
    that appeared on an unrelated crate or in nextest's prose, so the lines
    are selected by crate name and returned for a caller to inspect.

    >>> crate_compilations("  Running `rustc --crate-name probe`", "probe")
    ['Running `rustc --crate-name probe`']
    >>> crate_compilations("  Running `rustc --crate-name other`", "probe")
    []
    """
    return [
        line.strip()
        for line in stderr.splitlines()
        if "rustc" in line and f"--crate-name {crate}" in line
    ]


def harness_compilations(stderr: str, crate: str) -> list[str]:
    """Return the subset of compiler lines that built a test harness.

    A crate compiled both as a binary and as its test harness produces two
    lines carrying the same crate name, so the caller asserting on the test
    build has to select the one with Cargo's test marker.

    >>> lines = ["rustc --crate-name p src/main.rs", "rustc --crate-name p --test"]
    >>> harness_compilations("\\n".join(lines), "p")[0].endswith("--test")
    True
    """
    return [
        line for line in crate_compilations(stderr, crate) if _has_test_marker(line)
    ]


def _has_test_marker(line: str) -> bool:
    """Report whether a compiler line builds the crate's test harness.

    The check is per token because ``--test`` is bare: a path or a
    ``--test-args`` flag inside the same line would satisfy a substring test
    without being Cargo's marker.
    """
    return TEST_HARNESS_MARKER in line.split()


def assert_probe_inputs_are_real() -> None:
    """Require each copied input to exist and still be read by the Makefile.

    The list is checked rather than trusted because the Makefile gains
    parse-time reads over time, and a stale list would let a probe pass with
    a pin no developer has. The greeting is checked against the crate name for
    the same reason: an assertion reading one while the binary prints the other
    would fail as though the route were broken.
    """
    for relative in PROBE_INPUTS:
        assert (REPO_ROOT / relative).is_file(), (
            f"the probe copies {relative}, which the repository does not ship"
        )
    assert f'println!("{PROBE_GREETING}")' in PROBE_MAIN, (
        f"the probe source must print {PROBE_GREETING!r}, the greeting its "
        "callers assert on"
    )
    assert f'name = "{PROBE_CRATE}"' in PROBE_MANIFEST, (
        f"the probe manifest must name the crate {PROBE_CRATE!r}, which is how "
        "callers select its compiler lines"
    )


def nextest_missing_reason() -> str | None:
    """Return why ``dev-test`` cannot run here, or ``None`` when it can."""
    if shutil.which("cargo-nextest") is None:
        return "dev-test needs cargo-nextest, which scripts/install-nextest.sh installs"
    return None


def linker_missing_reason() -> str | None:
    """Return why a development target cannot link here, or ``None`` when it can.

    On Linux both development recipes add ``-Clink-arg=-fuse-ld=mold``, so a
    host without ``ld.mold`` fails in the linker rather than skipping. The
    recipes reach it through ``$(HOME)/.local/bin``, and that directory is
    what ``make install-dev-fast`` populates.

    ``shutil.which`` is the right test rather than a path join, because the
    recipe resolves the linker through ``PATH``: a developer who keeps it
    somewhere else is provisioned too, and the contract should run. Other
    hosts add no such flag and need nothing.
    """
    if sys.platform != "linux":
        return None
    if shutil.which("ld.mold") is None:
        return (
            "the development route links with ld.mold, which "
            "make install-dev-fast installs"
        )
    return None
