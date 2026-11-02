"""Contract-test how CI obtains its tools.

CI must never compile a tool from source: a source build turns a cache miss
into minutes of paid compilation and defeats any attempt to reason about
runner cost. Every tool therefore arrives as a pinned, checksum-verified
prebuilt archive, through a shared action pinned to one reviewed commit, or
through the shared ``rust-prover-tools`` package, which owns its own bounded
downloader and digest pins. These tests also pin the ordering that makes those
installs meaningful, because a tool installed after its first use is a tool the
job never had.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import collections.abc as cabc
import re
import typing as typ
from pathlib import Path

import pytest
from workflow_support import (
    ROOT,
    job,
    is_reusable_call,
    jobs,
    load_workflow,
    run_script,
    steps,
    uses_reference,
    workflow_names,
    workflow_paths,
)

#: Forms that build a tool from source, or that resolve a binary without a
#: reviewed pin. `cargo install` compiles; bare `cargo binstall` resolves a
#: version at run time and silently falls back to compiling.
FORBIDDEN_INSTALL_PATTERNS = (
    (re.compile(r"\bcargo\s+install\b"), "cargo install compiles from source"),
    (
        re.compile(r"\bcargo\s+binstall\b"),
        "cargo binstall resolves an unpinned binary and can fall back to a "
        "source build",
    ),
)

#: Every third-party install action must name a 40-hex commit, never a tag.
UNPINNED_USES_RE = re.compile(r"^[^@]+@(?!\b[0-9a-f]{40}\b)")

#: How a step provides a tool: by running a repository installer script, by
#: using a shared installer action, or by invoking a Make target that delegates
#: to the shared prover tool.
SCRIPT = "script"
ACTION = "action"
TARGET = "target"

#: Each installer and the first command that needs the tool it provides.
#:
#: Kani and Verus are targets rather than scripts because the download,
#: checksum, and extraction mechanics live in ``rust-prover-tools``; the
#: repository owns only the pins under ``tools/`` and the delegation.
INSTALLER_ORDER = (
    (SCRIPT, "scripts/install-nextest.sh", re.compile(r"cargo\s+nextest\b")),
    (SCRIPT, "scripts/install-sccache.sh", re.compile(r"\bsccache\s+--")),
    (TARGET, "install-kani", re.compile(r"\bmake\s+kani(-full)?\b")),
    (TARGET, "install-verus", re.compile(r"\bmake\s+verus\b")),
    (ACTION, "install-whitaker", re.compile(r"\bmake\s+lint\b")),
)

#: Tool pins that must exist as a version file and a digest manifest. Kani is
#: not among them: its two archive digests live in the pinned
#: ``rust-prover-tools`` release, and a local copy would be a second pin that
#: nothing reads and nothing keeps in step.
PINNED_TOOLS = ("nextest", "sccache", "verus")

#: Installer scripts a workflow invokes directly, so the committed file mode
#: is part of the contract: a non-executable script fails the job with
#: "Permission denied" rather than anything diagnosable.
INSTALLER_SCRIPTS = tuple(name for kind, name, _ in INSTALLER_ORDER if kind == SCRIPT)

#: Tools whose acquisition moved into ``rust-prover-tools``. A repository
#: script with one of these names would be a second downloader, which is the
#: duplicate the shared package exists to remove.
DELEGATED_TOOLS = ("install-kani.sh", "install-verus.sh", "run-verus.sh")

ROOT_MAKEFILE = ROOT / "Makefile"

#: `git+https://host/owner/repo@<40-hex commit>`. A branch or tag can move.
PROVER_TOOLS_SOURCE_RE = re.compile(
    r"^PROVER_TOOLS_SOURCE\s*\?=\s*git\+https://\S+@(?P<ref>[0-9a-f]{40})$",
    re.MULTILINE,
)


def _provides_tool(step: dict[str, typ.Any], kind: str, name: str) -> bool:
    """Report whether a step installs the named tool."""
    if kind == SCRIPT:
        return name in run_script(step)
    if kind == TARGET:
        return re.search(rf"\bmake\s+{re.escape(name)}\b", run_script(step)) is not None
    return f"/{name}@" in uses_reference(step)


def _first_step(
    job_steps: list[dict[str, typ.Any]],
    predicate: cabc.Callable[[dict[str, typ.Any]], bool],
) -> int | None:
    """Return the index of the first step satisfying the predicate."""
    return next(
        (index for index, step in enumerate(job_steps) if predicate(step)), None
    )


def _target_recipe(target: str) -> str:
    """Return one Makefile target's recipe body, without its prerequisite line.

    The recipe is read from the `target:` line to the next line that starts in
    column zero, so a target's body cannot silently absorb the next target's
    commands.
    """
    text = ROOT_MAKEFILE.read_text(encoding="utf-8").splitlines()
    starts = [
        index
        for index, line in enumerate(text)
        if re.match(rf"^{re.escape(target)}:", line)
    ]
    assert len(starts) == 1, (
        f"the Makefile must define exactly one {target}: target, "
        f"found {len(starts)}"
    )
    start = starts[0]
    body = [text[start]]
    for line in text[start + 1 :]:
        # An indented line is recipe or continuation; a line at column zero is
        # the next target, which ends this body.
        if line and not line[0].isspace():
            break
        body.append(line)
    return "\n".join(body)


@pytest.mark.parametrize("workflow_path", workflow_paths(), ids=workflow_names())
def test_no_workflow_builds_a_tool_from_source(workflow_path: Path) -> None:
    """Reject source-build and unpinned-binary install forms outright."""
    text = workflow_path.read_text(encoding="utf-8")
    for pattern, reason in FORBIDDEN_INSTALL_PATTERNS:
        assert not pattern.search(text), (
            f"{workflow_path.name} uses a forbidden install form: {reason}"
        )


@pytest.mark.parametrize("tool", PINNED_TOOLS)
def test_every_pinned_tool_has_a_version_and_a_digest(tool: str) -> None:
    """A pinned version without a digest is not a verified download."""
    version_file = ROOT / "tools" / tool / "VERSION"
    checksum_file = ROOT / "tools" / tool / "SHA256SUMS"
    assert version_file.is_file(), f"{version_file} must pin a version"
    version = version_file.read_text(encoding="utf-8").strip()
    assert version, f"{version_file} must not be empty"
    assert checksum_file.is_file(), f"{checksum_file} must pin a digest"
    digests = checksum_file.read_text(encoding="utf-8").splitlines()
    entries = [line.split() for line in digests if line.strip()]
    assert entries, f"{checksum_file} must list at least one archive"
    for digest, archive in ((line[0], line[1]) for line in entries):
        assert re.fullmatch(r"[0-9a-f]{64}", digest), (
            f"{checksum_file} entry for {archive} is not a SHA-256 digest"
        )
        assert version in archive, (
            f"{checksum_file} pins {archive}, which does not carry the "
            f"pinned version {version}"
        )


def test_kani_keeps_one_digest_owner() -> None:
    """Kani's two archives are pinned once, by the shared prover tool.

    Kani is a multi-part tool: extracting only the verifier bundle leaves
    ``cargo kani`` unavailable, and installing only the front-end leaves the
    verifier missing, so both archives are pinned together. ``rust-prover-tools``
    now holds both digests beside the version it supports and refuses a version
    it has no manifest for. A local ``tools/kani/SHA256SUMS`` would be a second
    pin that no command reads, so a Kani bump could update one and silently
    leave the other disagreeing.
    """
    version_file = ROOT / "tools" / "kani" / "VERSION"
    assert version_file.is_file(), f"{version_file} must pin the Kani version"
    stray = ROOT / "tools" / "kani" / "SHA256SUMS"
    assert not stray.exists(), (
        f"{stray} must not exist; the Kani archive digests are owned by the "
        "pinned rust-prover-tools release, and this repository reads neither"
    )


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_every_action_reference_is_pinned_to_a_commit(workflow_name: str) -> None:
    """A tag can be repointed; a commit SHA cannot."""
    workflow = load_workflow(workflow_name)
    for job_name, definition in jobs(workflow).items():
        references = [definition["uses"]] if is_reusable_call(definition) else []
        references += [
            reference
            for step in steps(definition)
            if (reference := uses_reference(step))
        ]
        for reference in references:
            assert not UNPINNED_USES_RE.match(reference), (
                f"{workflow_name}:{job_name} references {reference} without a "
                "40-character commit SHA"
            )


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_installers_run_before_the_tools_they_provide(workflow_name: str) -> None:
    """A tool installed after its first use was never actually installed."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        job_steps = steps(definition)
        for kind, name, consumer in INSTALLER_ORDER:
            uses_at = _first_step(
                job_steps, lambda step, c=consumer: bool(c.search(run_script(step)))
            )
            if uses_at is None:
                continue
            installs_at = _first_step(
                job_steps, lambda step, k=kind, n=name: _provides_tool(step, k, n)
            )
            assert installs_at is not None, (
                f"{workflow_name}:{job_name} runs {consumer.pattern} without "
                f"first installing {name}"
            )
            assert installs_at < uses_at, (
                f"{workflow_name}:{job_name} installs {name} after its first use"
            )


@pytest.mark.parametrize("installer", INSTALLER_SCRIPTS)
def test_installer_scripts_are_executable(installer: str) -> None:
    """A workflow runs these by path, so the mode bit is load-bearing."""
    path = ROOT / installer
    assert path.is_file(), f"{installer} must exist"
    assert path.stat().st_mode & 0o111, (
        f"{installer} is invoked directly by a workflow and must be executable"
    )


@pytest.mark.parametrize("script", DELEGATED_TOOLS)
def test_the_shared_prover_tool_replaced_the_bespoke_script(script: str) -> None:
    """Prover acquisition has one owner, so the local downloader must be gone.

    A retained script would be a second downloader beside
    ``rust-prover-tools``: a second place to update a pin, a second set of
    transfer limits, and a second answer to what a verified archive is.
    """
    assert not (ROOT / "scripts" / script).exists(), (
        f"scripts/{script} must be deleted; prover installation and execution "
        "delegate to rust-prover-tools"
    )


def test_the_prover_tool_pin_is_an_immutable_commit() -> None:
    """A branch or tag can be repointed; the installer's commit cannot."""
    match = PROVER_TOOLS_SOURCE_RE.search(
        ROOT_MAKEFILE.read_text(encoding="utf-8"),
    )
    assert match is not None, (
        "the Makefile must define PROVER_TOOLS_SOURCE as "
        "git+https://…@<40-character commit SHA>"
    )


@pytest.mark.parametrize(
    ("target", "command"),
    [
        ("install-kani", "kani install"),
        ("install-verus", "verus install"),
        ("verus", "verus run"),
        ("kani-check", "kani check-version"),
    ],
)
def test_prover_targets_delegate_to_the_shared_tool(
    target: str, command: str
) -> None:
    """Each prover target runs one `prover-tools` subcommand and nothing else.

    The Makefile is an infrastructure adapter. Asserting the subcommand keeps
    the mechanical work in the shared package while leaving the runner
    parameters, such as the proof list, free to change here.
    """
    recipe = _target_recipe(target)
    assert f"$(PROVER_TOOLS) {command}" in recipe, (
        f"the Makefile's {target} target must run `$(PROVER_TOOLS) {command}`, "
        f"got {recipe!r}"
    )
    assert "cargo install" not in recipe, (
        f"the Makefile's {target} target must not compile a tool from source"
    )


def test_the_kani_job_does_not_force_warning_denial() -> None:
    """`cfg(kani)` leaves proof helpers partly unused, so denial breaks it.

    Setup Rust exports `-D warnings` by default. Under `cargo kani` that
    turns expected dead-code warnings in the harness helpers into
    compilation errors, which is how the suite failed the first time it ever
    got far enough to compile.
    """
    definition = job("nightly-kani.yml", "kani-full")
    setup = next(
        step
        for step in steps(definition)
        if "/actions/setup-rust@" in uses_reference(step)
    )
    assert setup.get("with", {}).get("rustflags") == "", (
        "nightly-kani must pass an empty rustflags to Setup Rust so the "
        "verification build is not compiled with -D warnings"
    )
