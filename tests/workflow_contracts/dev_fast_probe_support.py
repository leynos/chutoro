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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The probe crate the real recipes build. It is deliberately minimal: the
#: subject is the Make recipe, not the crate, and a tiny tree keeps the
#: compiles inside a test-sized budget. The empty ``[workspace]`` stops the
#: probe joining the checkout's workspace if it is ever written inside it.
PROBE_MANIFEST = """\
[package]
name = "devprobe"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "devprobe"
path = "src/main.rs"

[workspace]
"""

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


def _rustup_home() -> Path | None:
    """Return the shared toolchain root, or ``None`` when rustup cannot say."""
    reported = subprocess.run(
        ["rustup", "show", "home"],
        check=False,
        capture_output=True,
        text=True,
    )
    if reported.returncode != 0 or not reported.stdout.strip():
        return None
    return Path(reported.stdout.strip())


def run_development_target(
    directory: Path, environment: dict[str, str], target: str
) -> subprocess.CompletedProcess[str]:
    """Run one development recipe against the probe crate in ``directory``.

    ``APP=devprobe`` points the recipe at the probe's binary and
    ``BUILD_JOBS=--verbose`` asks Cargo to print the compiler arguments the
    backend assertion reads. ``CI`` is cleared so the nextest profile is the
    developer's default rather than the gate's.

    Cargo writes its progress and nextest its run report to standard error,
    leaving standard output for make's echo of the recipe, so callers read the
    stream each assertion belongs to.
    """
    environment = dict(environment)
    environment.pop("CI", None)
    return subprocess.run(
        [
            "make",
            "--no-print-directory",
            "--file",
            str(directory / "Makefile"),
            target,
            "APP=devprobe",
            "BUILD_JOBS=--verbose",
        ],
        cwd=directory,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def assert_probe_inputs_are_real() -> None:
    """Require each copied input to exist and still be read by the Makefile.

    The list is checked rather than trusted because the Makefile gains
    parse-time reads over time, and a stale list would let a probe pass with
    a pin no developer has.
    """
    for relative in PROBE_INPUTS:
        assert (REPO_ROOT / relative).is_file(), (
            f"the probe copies {relative}, which the repository does not ship"
        )


def nextest_missing_reason() -> str | None:
    """Return why ``dev-test`` cannot run here, or ``None`` when it can."""
    if shutil.which("cargo-nextest") is None:
        return "dev-test needs cargo-nextest, which scripts/install-nextest.sh installs"
    return None
