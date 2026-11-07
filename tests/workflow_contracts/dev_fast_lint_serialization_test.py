"""Contract test for the lint gate's sequential Clippy and Whitaker halves.

`make lint` orders three checks: `interrogate` first, then Clippy, then the
Whitaker Dylint suite. Clippy and Whitaker share the package cache and the
compiler-admission pool, so `.NOTPARALLEL: lint` exists to keep them apart even
when make is invoked with parallel jobs. This module pins that ordering and the
serialization with controlled probes, so a recipe edit cannot put the two
competing for a compile slot. The remaining development-route contracts live in
`dev_fast_ci_contract_test`.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

from dev_fast_ci_contract_test import MAKEFILE_PATH, ROOT


def _make_probe(path: Path, role: str) -> None:
    """Write a controlled fake gate that records overlap with its peer."""
    source = textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        import os
        from pathlib import Path
        import time

        state = Path(os.environ["DEV_FAST_PROBE_STATE"])
        role = {role!r}
        peer = "whitaker" if role == "cargo" else "cargo"
        lock = state / "lock"

        def acquire_lock():
            deadline = time.monotonic() + 5
            while True:
                try:
                    lock.mkdir()
                    return
                except FileExistsError:
                    if time.monotonic() >= deadline:
                        raise SystemExit("timed out acquiring probe lock")
                    time.sleep(0.005)

        acquire_lock()
        if (state / f"{{peer}}.running").exists():
            (state / "overlap").write_text(f"{{role}} overlapped {{peer}}\\n")
        (state / f"{{role}}.running").write_text("running\\n")
        lock.rmdir()

        time.sleep(0.15)

        acquire_lock()
        (state / f"{{role}}.running").unlink()
        (state / f"{{role}}.ran").write_text("ran\\n")
        lock.rmdir()
        """
    )
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _make_interrogate_probe(path: Path, state: Path, *, exit_status: int) -> None:
    """Write a fake `interrogate` that records its start and exits as asked.

    The recipe runs the checker through `$(INTERROGATE)`, so this stands in for
    the `uv`-backed command and keeps the test off the network. The exit status
    is a parameter because the two contracts need opposite answers: the
    ordering check must reach both lint halves, and the failure check must stop
    the run before either of them.
    """
    path.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            from pathlib import Path

            state = Path({str(state)!r})
            (state / "interrogate.ran").write_text("ran\\n")
            print("interrogate probe")
            raise SystemExit({exit_status})
            """
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


def _run_lint(
    tmp_path: Path, job_count: str, *, interrogate_exit: int
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run the lint target against probes and return the result with its state."""
    cargo = tmp_path / "probe-cargo"
    whitaker = tmp_path / "probe-whitaker"
    interrogate = tmp_path / "probe-interrogate"
    state = tmp_path / "state"
    state.mkdir()
    _make_probe(cargo, "cargo")
    _make_probe(whitaker, "whitaker")
    _make_interrogate_probe(interrogate, state, exit_status=interrogate_exit)

    environment = os.environ.copy()
    environment["DEV_FAST_PROBE_STATE"] = str(state)
    completed = subprocess.run(
        [
            "make",
            "--no-print-directory",
            job_count,
            "--file",
            str(MAKEFILE_PATH),
            "lint",
            f"CARGO={cargo}",
            f"WHITAKER={whitaker}",
            f"INTERROGATE={interrogate}",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    return completed, state


def test_parallel_lint_serializes_clippy_and_whitaker(tmp_path: Path) -> None:
    """`make -j lint` must still run Clippy and Whitaker one after the other.

    Both halves share the package cache and the compiler-admission pool, so
    overlap is what `.NOTPARALLEL: lint` exists to prevent.
    """
    completed, state = _run_lint(tmp_path, "-j2", interrogate_exit=0)

    assert completed.returncode == 0, (
        f"controlled lint probe failed: {completed.stdout}\n{completed.stderr}"
    )
    # The absence of an overlap marker only means something if both probes
    # actually ran. A recipe that skipped `lint-clippy` or `lint-whitaker`
    # would trivially never overlap, so record each role's completion and
    # require both.
    for role in ("cargo", "whitaker"):
        assert (state / f"{role}.ran").exists(), (
            f"lint did not run the {role} probe under make -j2: "
            f"{sorted(path.name for path in state.iterdir())}"
        )
    assert not (state / "overlap").exists(), (
        "lint-clippy and lint-whitaker ran concurrently under make -j2"
    )


def test_a_failing_interrogate_stops_the_lint_run(tmp_path: Path) -> None:
    """Interrogate runs first, so its failure must stop both lint halves.

    Budgeting for a possibly unmocked docstring gate is the wrong fix: it would
    also hide a real regression that makes `make lint` continue past a failed
    check. Interrogate is stubbed instead, and this pins why the stub is honest.
    """
    completed, state = _run_lint(tmp_path, "-j1", interrogate_exit=1)

    assert completed.returncode != 0, (
        f"lint must fail when interrogate fails: {completed.stdout}"
    )
    assert (state / "interrogate.ran").exists(), (
        "lint did not run the interrogate probe at all"
    )
    for role in ("cargo", "whitaker"):
        assert not (state / f"{role}.ran").exists(), (
            f"lint ran the {role} probe after interrogate failed: "
            f"{sorted(path.name for path in state.iterdir())}"
        )
