"""Contract-test who owns each cached path.

A mutable cache path with two owners has two keys racing to describe it, so
one of them always restores work the other has already invalidated. Every
path below therefore has exactly one owner per job, and every key is
derived from a correctness input a reader can explain. Compiler output is
excluded outright: sccache owns it, and archiving a ``target`` tree beside
sccache duplicates that ownership and inflates the weekly quota.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import collections
import re
from pathlib import Path

import pytest
from workflow_support import (
    CACHE_ACTION_SHA,
    SHARED_ACTION_OWNED_PATHS,
    cache_is_enabled,
    declared_cache_paths,
    job,
    jobs,
    load_workflow,
    shared_action_name,
    steps,
    uses_reference,
    workflow_names,
    workflow_paths,
)

CACHE_ACTION_RE = re.compile(r"^actions/cache(?:/(?:restore|save))?@(?P<sha>\S+)$")

#: setup-uv owns ~/.cache/uv whenever its cache is enabled, wherever it is
#: invoked from.
UV_CACHE_PATH = "~/.cache/uv"

#: The three parts of a Kani installation. Restoring fewer than all three
#: leaves either `cargo kani` missing or the verifier toolchain symlink
#: dangling, so they form one cache generation.
KANI_CACHE_PATHS = frozenset(
    {"~/.cargo/bin/cargo-kani", "~/.cargo/bin/kani", "~/.kani", "~/.kani-rustup"}
)


def _cache_steps(definition: dict) -> list[dict]:
    """Return the job's explicit cache steps."""
    return [
        step
        for step in steps(definition)
        if CACHE_ACTION_RE.match(uses_reference(step))
    ]


def _owned_paths(definition: dict) -> list[tuple[str, str]]:
    """Return every (path, owner) pair a job establishes."""
    owners: list[tuple[str, str]] = []
    for step in steps(definition):
        reference = uses_reference(step)
        if CACHE_ACTION_RE.match(reference):
            label = step.get("name", reference)
            owners += [(path, label) for path in declared_cache_paths(step)]
            continue
        if reference.startswith("astral-sh/setup-uv@") and cache_is_enabled(step):
            owners.append((UV_CACHE_PATH, "astral-sh/setup-uv"))
            continue
        action = shared_action_name(reference)
        if action in SHARED_ACTION_OWNED_PATHS and cache_is_enabled(step):
            owners += [(path, action) for path in SHARED_ACTION_OWNED_PATHS[action]]
    return owners


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_every_cache_step_uses_the_pinned_cache_action(workflow_name: str) -> None:
    """One reviewed cache implementation, everywhere."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        for step in _cache_steps(definition):
            match = CACHE_ACTION_RE.match(uses_reference(step))
            assert match is not None
            assert match.group("sha") == CACHE_ACTION_SHA, (
                f"{workflow_name}:{job_name} pins the cache action at "
                f"{match.group('sha')}, not {CACHE_ACTION_SHA}"
            )


@pytest.mark.parametrize("workflow_path", workflow_paths(), ids=workflow_names())
def test_no_workflow_uses_the_ubicloud_cache_fork(workflow_path: Path) -> None:
    """The transparent proxy intercepts actions/cache, so the fork is dead."""
    assert "ubicloud/cache" not in workflow_path.read_text(encoding="utf-8"), (
        f"{workflow_path.name} must use actions/cache, not the deprecated "
        "ubicloud/cache fork"
    )


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_no_cache_step_archives_a_target_tree(workflow_name: str) -> None:
    """sccache is the sole owner of compiler output."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        for step in _cache_steps(definition):
            for path in declared_cache_paths(step):
                assert not re.search(r"(^|/)target(/|$)", path), (
                    f"{workflow_name}:{job_name} archives {path}; sccache owns "
                    "compiler output and target trees must not be cached"
                )


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_each_cached_path_has_exactly_one_owner(workflow_name: str) -> None:
    """Two owners for one path means two keys racing to describe it."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        counts = collections.Counter(path for path, _ in _owned_paths(definition))
        duplicated = {path: count for path, count in counts.items() if count > 1}
        assert not duplicated, (
            f"{workflow_name}:{job_name} gives these paths more than one cache "
            f"owner: {sorted(duplicated)}"
        )


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_every_cache_step_declares_an_explainable_key(workflow_name: str) -> None:
    """A cache without a key cannot explain a miss."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        for step in _cache_steps(definition):
            with_block = step.get("with", {})
            key = with_block.get("key")
            assert isinstance(key, str) and key.strip(), (
                f"{workflow_name}:{job_name} declares a cache step without a key"
            )


def test_kani_restores_its_front_end_bundle_and_toolchain_together() -> None:
    """Pin Kani's multi-part cache contract.

    The front-end lives in Cargo home, the verifier bundle under KANI_HOME,
    and the pinned nightly toolchain in a Kani-specific rustup home that the
    bundle symlinks into. Caching any subset leaves a warm run either
    without ``cargo kani`` or with a dangling toolchain link.
    """
    definition = job("nightly-kani.yml", "kani-full")
    cache_steps = _cache_steps(definition)
    assert len(cache_steps) == 1, "the Kani job must declare exactly one cache step"
    paths = set(declared_cache_paths(cache_steps[0]))
    assert paths == KANI_CACHE_PATHS, (
        f"the Kani cache must cover {sorted(KANI_CACHE_PATHS)}, found {sorted(paths)}"
    )
    key = cache_steps[0]["with"]["key"]
    for pin_file in ("tools/kani/VERSION", "tools/kani/SHA256SUMS"):
        assert pin_file in key, (
            f"the Kani cache key must derive from {pin_file} so a pin bump "
            "invalidates it"
        )


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_cache_setup_precedes_the_install_it_protects(workflow_name: str) -> None:
    """A cache restored after an install has already paid for the install."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        job_steps = steps(definition)
        first_cache = next(
            (
                index
                for index, step in enumerate(job_steps)
                if CACHE_ACTION_RE.match(uses_reference(step))
            ),
            None,
        )
        if first_cache is None:
            continue
        first_install = next(
            (
                index
                for index, step in enumerate(job_steps)
                if "scripts/install-" in (step.get("run") or "")
            ),
            None,
        )
        if first_install is None:
            continue
        assert first_cache < first_install, (
            f"{workflow_name}:{job_name} restores its cache after the install "
            "it is meant to avoid"
        )
