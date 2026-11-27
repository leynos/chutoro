"""Contract tests for the supply-chain gate, its CI job, and its policy.

`make deny` is only as strong as the recipe behind it, the workflow that runs
it, and the policy file it reads. A target that echoes the command instead of
running it, a CI job that cannot fail the pull request, or a policy that lists
a licence without saying why all look like a gate from a distance. These tests
run the target against a recording stub so the argv and the failure that reach
the tool are observed rather than asserted as recipe text.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import collections.abc as cabc
import os
import re
import subprocess
import tomllib
import typing as typ
from pathlib import Path

import pytest
from makefile_reading import phony_targets
from scheduling_recipe_reading import recipe_commands
from workflow_support import ROOT, job, run_script, steps, uses_reference

MAKEFILE = ROOT / "Makefile"
DENY_CONFIG = ROOT / "deny.toml"
CI_WORKFLOW = "ci.yml"
CI_JOB = "cargo-deny"
DENY_RECIPE = "$(CARGO) deny --locked --config deny.toml check advisories bans licenses"
#: The licences the resolved graph requires. Adding one is a policy change,
#: so this list is asserted exactly rather than as a minimum.
REQUIRED_LICENCES = frozenset(
    {
        "Apache-2.0",
        "Apache-2.0 WITH LLVM-exception",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC0-1.0",
        "CDLA-Permissive-2.0",
        "ISC",
        "MIT",
        "Unicode-3.0",
        "Zlib",
    }
)
PINNED_CHECKOUT = re.compile(r"^actions/checkout@[0-9a-f]{40}$")
PINNED_SETUP_RUST = re.compile(
    r"^leynos/shared-actions/\.github/actions/setup-rust@[0-9a-f]{40}$"
)


def _makefile_text() -> str:
    """Return the root Makefile's text."""
    return MAKEFILE.read_text(encoding="utf-8")


def _policy() -> dict[str, typ.Any]:
    """Return `deny.toml` parsed as TOML, which also proves it is valid."""
    return tomllib.loads(DENY_CONFIG.read_text(encoding="utf-8"))


def _comment_block_before(lines: list[str], index: int) -> list[str]:
    """Return the unbroken comment lines immediately above `lines[index]`.

    A policy entry without a comment directly above it is the shape this
    helper exists to reject, so the walk stops at the first non-comment line.
    """
    block: list[str] = []
    cursor = index - 1
    while cursor >= 0 and lines[cursor].lstrip().startswith("#"):
        block.insert(0, lines[cursor])
        cursor -= 1
    return block


def _line_index(lines: list[str], needle: str) -> int:
    """Return the index of the first line containing `needle`.

    A missing needle fails the contract with the text it looked for, because
    the bare `StopIteration` a generator would raise says nothing about which
    policy rule the test expected to find.
    """
    for index, line in enumerate(lines):
        if needle in line:
            return index
    message = f"deny.toml has no line containing {needle!r}"
    raise AssertionError(message)


def _step_index(
    job_steps: list[dict[str, typ.Any]],
    predicate: cabc.Callable[[dict[str, typ.Any]], bool],
) -> int:
    """Return the index of the first step satisfying `predicate`.

    A job that has lost a step is a contract failure, and the step names are
    what makes the failure readable, so they are reported rather than a bare
    `StopIteration`.
    """
    for index, step in enumerate(job_steps):
        if predicate(step):
            return index
    names = [step.get("name") or step.get("uses") for step in job_steps]
    message = f"no step matched; the job's steps are {names}"
    raise AssertionError(message)


def _runs(step: dict[str, typ.Any], command: str) -> bool:
    """Report whether a step's `run` script is exactly `command`."""
    return run_script(step).strip() == command


def test_deny_is_a_phony_target() -> None:
    """A same-named file must not be able to skip the gate."""
    assert "deny" in phony_targets(_makefile_text())


def test_deny_runs_the_three_checks_and_nothing_else() -> None:
    """The recipe must run the global options before `check`, unmasked."""
    assert recipe_commands(_makefile_text(), "deny") == [DENY_RECIPE]


def test_all_runs_the_supply_chain_gate() -> None:
    """`make all` must not build and test while skipping the policy check."""
    header = next(
        (line for line in _makefile_text().splitlines() if line.startswith("all:")),
        None,
    )
    assert header is not None, "the Makefile must declare an all: target"
    prerequisites = set(header.split(":", 1)[1].split("##", 1)[0].split())
    assert {"release", "spelling", "test-workflow-contracts", "deny"} <= (prerequisites)


def _ignore_problem(entry: object) -> str | None:
    """Return why one advisory ignore breaks the policy, or ``None``.

    The documented procedure allows an ignore only as a note with a bound,
    so a bare string, a missing reason, or any entry without an expiry is a
    hole the gate would otherwise never report. cargo-deny accepts `expiry`
    on both advisory-id and crate-selector entries.

    Examples
    --------
    >>> _ignore_problem({"id": "RUSTSEC-0000-0000", "reason": "x", "expiry": "2026-12-31"})
    >>> _ignore_problem({"crate": "example@1.0.0", "reason": "x", "expiry": "2026-12-31"})
    >>> _ignore_problem({"id": "RUSTSEC-0000-0000", "reason": "x"}) is None
    False
    >>> _ignore_problem({"crate": "example@1.0.0", "reason": "x"}) is None
    False
    >>> _ignore_problem("RUSTSEC-0000-0000") is None
    False
    """
    if not isinstance(entry, dict):
        return f"{entry!r} must be a table carrying a reason"
    if not entry.get("reason"):
        return f"{entry!r} must carry a reason"
    # cargo-deny 0.22.1 accepts `expiry` on a crate selector as well as on an
    # advisory id, so the documented rule can hold for every entry.
    if not entry.get("expiry"):
        return f"{entry!r} must carry an expiry"
    if "id" not in entry and "crate" not in entry:
        return f"{entry!r} must name an advisory id or a crate"
    return None


def test_policy_blocks_advisories_and_yanked_crates() -> None:
    """The advisory gate must deny yanked crates and bound every ignore."""
    advisories = _policy()["advisories"]
    assert advisories["unmaintained"] == "workspace"
    assert advisories["yanked"] == "deny"
    problems = [
        problem
        for entry in advisories["ignore"]
        if (problem := _ignore_problem(entry)) is not None
    ]
    assert not problems, problems


def test_policy_covers_every_feature_and_dev_edge() -> None:
    """A dev-only dependency is still built by CI and must be in scope."""
    graph = _policy()["graph"]
    assert graph["all-features"] is True
    assert "targets" not in graph
    assert graph.get("exclude-dev") is not True


def test_policy_allows_exactly_the_licences_the_graph_requires() -> None:
    """The allowlist is a policy decision, so an addition must be deliberate."""
    licenses = _policy()["licenses"]
    assert frozenset(licenses["allow"]) == REQUIRED_LICENCES
    assert "ISC" in REQUIRED_LICENCES


def test_policy_bans_duplicate_digest_generations() -> None:
    """Every other duplicate warns; a second `digest` must fail the gate."""
    bans = _policy()["bans"]
    assert bans["multiple-versions"] == "warn"
    assert bans["multiple-versions-include-dev"] is True
    assert bans["wildcards"] == "deny"
    assert bans["allow-wildcard-paths"] is True
    entries = [entry for entry in bans["deny"] if entry.get("crate") == "digest"]
    assert len(entries) == 1
    assert entries[0]["deny-multiple-versions"] is True


def test_every_policy_exception_names_its_reason() -> None:
    """A rule a reader cannot justify is a rule that gets deleted blindly."""
    lines = DENY_CONFIG.read_text(encoding="utf-8").splitlines()
    digest_header = _line_index(lines, "[[bans.deny]]")
    digest_comment = " ".join(_comment_block_before(lines, digest_header))
    assert "digest" in digest_comment
    assert "generation" in digest_comment

    # The needle deliberately omits the comment marker: activating the
    # documented exception removes `# ` from the header, and the rationale
    # must still be found immediately above it either way.
    bzip2_header = _line_index(lines, "[[licenses.exceptions]]")
    bzip2_comment = " ".join(_comment_block_before(lines, bzip2_header))
    assert "libbz2-rs-sys" in bzip2_comment
    assert "bzip2-1.0.6" in bzip2_comment


def test_ci_job_is_independent_blocking_and_read_only() -> None:
    """A supply-chain failure must block the pull request on its own lane."""
    definition = job(CI_WORKFLOW, CI_JOB)
    assert definition["runs-on"] == "ubuntu-latest"
    assert definition["permissions"] == {"contents": "read"}
    assert isinstance(definition["timeout-minutes"], int)
    assert definition["timeout-minutes"] <= 15
    for key in ("needs", "if", "continue-on-error"):
        assert key not in definition, f"the job must not set {key}"
    for step in steps(definition):
        for key in ("if", "continue-on-error"):
            assert key not in step, f"the job's steps must not set {key}"


def test_ci_job_installs_the_pinned_tool_before_using_it() -> None:
    """The job must run the repository installer, then the gate."""
    job_steps = steps(job(CI_WORKFLOW, CI_JOB))
    assert PINNED_CHECKOUT.match(uses_reference(job_steps[0]))
    setup = next(
        step for step in job_steps if PINNED_SETUP_RUST.match(uses_reference(step))
    )
    assert setup["with"]["use-sccache"] == "false"

    install_at = _step_index(
        job_steps,
        lambda step: "scripts/install-cargo-deny.sh" in run_script(step),
    )
    install = job_steps[install_at]
    # The installer must keep its default directory. `make` prepends
    # `${HOME}/.cargo/bin` to PATH for every recipe, so a binary installed
    # anywhere else is shadowed inside `make deny`, and the pinned version
    # check below would then vouch for a binary the gate does not run.
    assert "CARGO_DENY_INSTALL_DIR" not in (install.get("env") or {}), (
        "the job must not override the install directory; the Makefile's PATH "
        "would shadow it inside `make deny`"
    )
    assert run_script(install).strip() == "scripts/install-cargo-deny.sh"

    version_at = _step_index(
        job_steps, lambda step: "cargo deny --version" in run_script(step)
    )
    version_run = run_script(job_steps[version_at])
    assert "tools/cargo-deny/VERSION" in version_run, (
        "the version step must compare against the pin; printing whatever "
        "binary answered would accept a different cargo-deny on PATH"
    )
    gate_at = _step_index(job_steps, lambda step: _runs(step, "make deny"))
    assert install_at < version_at < gate_at


def _version_step_script() -> str:
    """Return the CI job's cargo-deny version-check script."""
    job_steps = steps(job(CI_WORKFLOW, CI_JOB))
    index = _step_index(
        job_steps, lambda step: "cargo deny --version" in run_script(step)
    )
    return run_script(job_steps[index])


def _run_version_step(
    tmp_path: Path, reported: str
) -> subprocess.CompletedProcess[str]:
    """Run the version-check script against a stub ``cargo`` reporting `reported`."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stub(
        bin_dir / "cargo",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf 'cargo-deny {reported}\\n'
""",
    )
    script = tmp_path / "version-step.sh"
    script.write_text(_version_step_script(), encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    return subprocess.run(
        ["bash", "-e", str(script)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.mark.parametrize("accepted", (True, False), ids=("matching", "mismatched"))
def test_version_step_rejects_a_binary_that_is_not_the_pin(
    tmp_path: Path, accepted: bool
) -> None:
    """The script must compare the answer, not merely print it.

    A script that ran `cargo deny --version` and printed the pin without
    comparing the two would satisfy a substring assertion while accepting
    any cargo-deny that happened to answer. Running the extracted script
    against a stub proves the comparison is load-bearing.
    """
    pinned = (
        (ROOT / "tools" / "cargo-deny" / "VERSION").read_text(encoding="utf-8").strip()
    )
    completed = _run_version_step(tmp_path, pinned if accepted else "0.20.0")

    if accepted:
        assert completed.returncode == 0, completed.stderr
    else:
        assert completed.returncode != 0
        assert "expected cargo-deny" in completed.stderr


#: Records each cargo-deny invocation, one argument per line, then exits with
#: the status the test asks for.
CARGO_DENY_STUB = r"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" >> "$STUB_LOG"
exit "${STUB_STATUS:-0}"
"""

#: Stands in for `cargo`, dropping the `deny` subcommand name and handing the
#: rest to the recording cargo-deny, as cargo itself would.
CARGO_STUB = r"""#!/usr/bin/env bash
set -euo pipefail
[[ "${1:-}" == deny ]] || {
  printf 'unexpected cargo invocation: %s\n' "$*" >&2
  exit 64
}
shift
exec "$STUB_CARGO_DENY" "$@"
"""


def _write_stub(path: Path, body: str) -> None:
    """Write an executable script at `path`."""
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


@pytest.mark.parametrize("status", (0, 7))
def test_make_deny_propagates_the_gate_status(tmp_path: Path, status: int) -> None:
    """Success stays success, and a failure is reported rather than masked.

    GNU make exits with 2 for a failed recipe and names the command's own
    status in its diagnostic, so `Error 7` is the evidence that the gate's
    failure reached make instead of being swallowed by the recipe.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stub(bin_dir / "cargo", CARGO_STUB)
    _write_stub(bin_dir / "cargo-deny", CARGO_DENY_STUB)
    log = tmp_path / "invocations"
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"MAKEFLAGS", "MFLAGS", "CARGO"}
    }
    env |= {
        "STUB_CARGO_DENY": str(bin_dir / "cargo-deny"),
        "STUB_LOG": str(log),
        "STUB_STATUS": str(status),
    }

    completed = subprocess.run(
        ["make", "--no-print-directory", f"CARGO={bin_dir / 'cargo'}", "deny"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert log.read_text(encoding="utf-8").splitlines() == [
        "--locked",
        "--config",
        "deny.toml",
        "check",
        "advisories",
        "bans",
        "licenses",
    ]
    if status == 0:
        assert completed.returncode == 0, completed.stderr
    else:
        assert completed.returncode != 0
        assert f"Error {status}" in completed.stderr, completed.stderr
