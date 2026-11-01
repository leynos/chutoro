"""Behavioural tests for the Make targets that delegate to the prover tool.

A recipe that merely mentions the shared tool proves nothing: it could echo the
command, or leave it in a comment and run ``true``. These tests run each target
with ``PROVER_TOOLS`` pointed at a recorder, so the argv that reaches the tool
and the exit status the target propagates are observed rather than asserted as
recipe text.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from workflow_support import ROOT

#: Records one invocation per line, arguments separated by tabs, then exits
#: with the status the test asks for.
RECORDER = r"""#!/bin/sh
{
  for arg in "$@"; do
    printf '%s\t' "$arg"
  done
  printf '\n'
} >> "$FAKE_PROVER_LOG"
exit "${FAKE_PROVER_STATUS:-0}"
"""

#: The proofs `make verus` runs when neither override is set, in order.
DEFAULT_PROOFS = ("verus/edge_harvest_proofs.rs", "verus/session_core_distance.rs")

#: The Cargo the recipes interpolate into `kani-check`'s command.
CARGO = "cargo"


@dataclass(frozen=True)
class Recorder:
    """A stand-in prover tool that records the argv it is given.

    The executable and the log it writes to must agree, so they are kept
    together rather than threaded through every call. The exit status is
    configuration too: a test that wants a failing verifier configures one
    instead of passing a status through each invocation.
    """

    tool: Path
    log: Path
    status: int = 0

    @classmethod
    def create(cls, directory: Path, status: int = 0) -> "Recorder":
        """Write the recorder into `directory` and describe it."""
        tool = directory / "prover-tools"
        tool.write_text(RECORDER, encoding="utf-8")
        tool.chmod(0o755)
        return cls(tool=tool, log=directory / "invocations.log", status=status)

    def run(self, target: str, *variables: str) -> subprocess.CompletedProcess[str]:
        """Run one Make target against the recorder.

        `CARGO` is set explicitly so the recorded `kani-check` argv does not
        depend on whether the ambient environment exports one. `MAKEFLAGS` is
        dropped so an inherited `-n` or `-j` cannot turn the run into a dry
        run, and the recipe variables are dropped so a developer's shell cannot
        change the argv these tests pin.
        """
        unset = {
            "MAKEFLAGS",
            "MFLAGS",
            "KANI_INSTALL_FLAGS",
            "VERUS_INSTALL_FLAGS",
            "VERUS_PROOF_FILE",
            "VERUS_PROOF_FILES",
        }
        env = {key: value for key, value in os.environ.items() if key not in unset}
        env |= {
            "FAKE_PROVER_LOG": str(self.log),
            "FAKE_PROVER_STATUS": str(self.status),
        }
        command = [
            "make",
            "--no-print-directory",
            f"PROVER_TOOLS={self.tool}",
            f"CARGO={CARGO}",
            target,
            *variables,
        ]
        return subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

    def invocations(self) -> list[list[str]]:
        """Return each recorded invocation's argv, in call order."""
        if not self.log.exists():
            return []
        lines = self.log.read_text(encoding="utf-8").splitlines()
        return [line.rstrip("\t").split("\t") for line in lines if line]


@pytest.mark.parametrize(
    ("target", "variables", "expected"),
    [
        ("install-kani", (), ["kani", "install"]),
        (
            "install-kani",
            ("KANI_INSTALL_FLAGS=--compile",),
            ["kani", "install", "--compile"],
        ),
        ("install-verus", (), ["verus", "install"]),
        (
            "install-verus",
            ("VERUS_INSTALL_FLAGS=--install-dir /tmp/verus",),
            ["verus", "install", "--install-dir", "/tmp/verus"],
        ),
        (
            "kani-check",
            (),
            ["kani", "check-version", "--kani-command", f"{CARGO} kani"],
        ),
    ],
)
def test_targets_pass_one_delegated_subcommand(
    tmp_path: Path,
    target: str,
    variables: tuple[str, ...],
    expected: list[str],
) -> None:
    """Each target runs the shared tool once, with the subcommand it names."""
    recorder = Recorder.create(tmp_path)
    completed = recorder.run(target, *variables)

    assert completed.returncode == 0, (
        f"make {target} failed: stdout={completed.stdout!r} "
        f"stderr={completed.stderr!r}"
    )
    assert recorder.invocations() == [expected], (
        f"make {target} must invoke the prover tool exactly as {expected!r}, "
        f"recorded {recorder.invocations()!r}"
    )


def test_verus_runs_the_default_proofs_in_order(tmp_path: Path) -> None:
    """The default proof list reaches the tool in repository order."""
    recorder = Recorder.create(tmp_path)
    completed = recorder.run("verus")

    assert completed.returncode == 0, completed.stderr
    (argv,) = recorder.invocations()
    forwarded = [
        argv[index + 1] for index, arg in enumerate(argv) if arg == "--proof-file"
    ]
    assert tuple(forwarded) == DEFAULT_PROOFS, (
        f"make verus must forward {DEFAULT_PROOFS!r} in that order, got "
        f"{forwarded!r}"
    )


@pytest.mark.parametrize(
    ("variable", "proofs"),
    [
        ("VERUS_PROOF_FILE=verus/focused_proof.rs", ("verus/focused_proof.rs",)),
        (
            "VERUS_PROOF_FILES=verus/first.rs verus/second.rs",
            ("verus/first.rs", "verus/second.rs"),
        ),
    ],
)
def test_verus_honours_the_proof_overrides(
    tmp_path: Path, variable: str, proofs: tuple[str, ...]
) -> None:
    """Both the single-file and the list override replace the default list."""
    recorder = Recorder.create(tmp_path)
    completed = recorder.run("verus", variable)

    assert completed.returncode == 0, completed.stderr
    (argv,) = recorder.invocations()
    forwarded = [
        argv[index + 1] for index, arg in enumerate(argv) if arg == "--proof-file"
    ]
    assert tuple(forwarded) == proofs, (
        f"make verus with {variable!r} must forward {proofs!r}, got {forwarded!r}"
    )


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("install-kani", "install-kani failed"),
        ("install-verus", "install-verus failed"),
        ("kani-check", "kani-check failed"),
        ("verus", "verus proofs failed"),
    ],
)
def test_targets_propagate_a_failing_tool(
    tmp_path: Path, target: str, message: str
) -> None:
    """A verifier failure must fail the build and name the status it saw.

    GNU make normalises a failed recipe to exit status 2, so what a caller can
    observe is that the build failed. What the contributor reading the job log
    needs is the status the tool itself chose: collapsing every failure to one
    fixed message would hide which phase failed, and would make a missing pin
    indistinguishable from a broken archive.
    """
    recorder = Recorder.create(tmp_path, status=7)
    completed = recorder.run(target)

    assert completed.returncode != 0, (
        f"make {target} must fail when the tool fails, got exit status "
        f"0; stdout={completed.stdout!r}"
    )
    assert f"{message} (exit 7)" in completed.stderr, (
        f"make {target} must report {message!r} with the tool's status to "
        f"stderr, got {completed.stderr!r}"
    )
    assert len(recorder.invocations()) == 1, (
        f"make {target} must not retry a failed invocation, recorded "
        f"{recorder.invocations()!r}"
    )