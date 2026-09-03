"""Contract-test how CI obtains its tools.

CI must never compile a tool from source: a source build turns a cache miss
into minutes of paid compilation and defeats any attempt to reason about
runner cost. Every tool therefore arrives as a pinned, checksum-verified
prebuilt archive, or through a shared action pinned to one reviewed commit.
These tests also pin the ordering that makes those installs meaningful,
because a tool installed after its first use is a tool the job never had.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from workflow_support import (
    ROOT,
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

#: Each installer script and the first command that needs its tool.
INSTALLER_ORDER = (
    ("scripts/install-nextest.sh", re.compile(r"cargo\s+nextest\b")),
    ("scripts/install-kani.sh", re.compile(r"\bmake\s+kani(-full)?\b")),
    ("scripts/install-verus.sh", re.compile(r"\bmake\s+verus\b")),
)

#: Shared actions whose installed tool a later step invokes.
SHARED_INSTALLER_ORDER = (("install-whitaker", re.compile(r"\bmake\s+lint\b")),)

#: Tool pins that must exist as a version file and a digest manifest.
PINNED_TOOLS = ("kani", "nextest", "verus")


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


def test_kani_pins_its_front_end_and_its_verifier_bundle() -> None:
    """Kani is a multi-part tool and both parts must be pinned together.

    Extracting only the verifier bundle leaves ``cargo kani`` unavailable;
    installing only the front-end leaves the verifier missing. The
    front-end derives the bundle's name from its own version, so a single
    version governs both archives.
    """
    version = (ROOT / "tools" / "kani" / "VERSION").read_text(encoding="utf-8").strip()
    archives = {
        line.split()[1]
        for line in (ROOT / "tools" / "kani" / "SHA256SUMS")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    assert f"kani-verifier-{version}-x86_64-unknown-linux-gnu.tar.gz" in archives, (
        "the cargo-kani front-end archive must be pinned"
    )
    assert f"kani-{version}-x86_64-unknown-linux-gnu.tar.gz" in archives, (
        "the Kani verifier bundle must be pinned"
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
        scripts = [run_script(step) for step in steps(definition)]
        for installer, consumer in INSTALLER_ORDER:
            uses_at = [
                index for index, script in enumerate(scripts) if consumer.search(script)
            ]
            if not uses_at:
                continue
            installs_at = [
                index for index, script in enumerate(scripts) if installer in script
            ]
            assert installs_at, (
                f"{workflow_name}:{job_name} runs {consumer.pattern} without "
                f"first running {installer}"
            )
            assert min(installs_at) < min(uses_at), (
                f"{workflow_name}:{job_name} runs {installer} after its first use"
            )


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_shared_installers_run_before_the_tools_they_provide(
    workflow_name: str,
) -> None:
    """Apply the same ordering rule to shared installer actions."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        job_steps = steps(definition)
        scripts = [run_script(step) for step in job_steps]
        references = [uses_reference(step) for step in job_steps]
        for action, consumer in SHARED_INSTALLER_ORDER:
            uses_at = [
                index for index, script in enumerate(scripts) if consumer.search(script)
            ]
            if not uses_at:
                continue
            installs_at = [
                index
                for index, reference in enumerate(references)
                if f"/{action}@" in reference
            ]
            assert installs_at, (
                f"{workflow_name}:{job_name} runs {consumer.pattern} without "
                f"first using the {action} action"
            )
            assert min(installs_at) < min(uses_at), (
                f"{workflow_name}:{job_name} uses {action} after its first use"
            )
