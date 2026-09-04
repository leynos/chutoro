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
import collections.abc as cabc
import re
from pathlib import Path

import pytest
from workflow_support import (
    CACHE_ACTION_SHA,
    SHARED_ACTION_OWNED_PATHS,
    cache_is_enabled,
    declared_cache_paths,
    jobs,
    load_workflow,
    shared_action_name,
    run_script,
    steps,
    uses_reference,
    workflow_names,
    workflow_paths,
)

CACHE_ACTION_RE = re.compile(r"^actions/cache(?:/(?:restore|save))?@(?P<sha>\S+)$")

#: setup-uv owns ~/.cache/uv whenever its cache is enabled, wherever it is
#: invoked from.
UV_CACHE_PATH = "~/.cache/uv"


def _cache_steps(definition: dict) -> list[dict]:
    """Return the job's explicit cache steps."""
    return [
        step
        for step in steps(definition)
        if CACHE_ACTION_RE.match(uses_reference(step))
    ]


def _explicit_owner(step: dict, reference: str) -> list[tuple[str, str]]:
    """Return the paths a cache step names outright."""
    label = step.get("name", reference)
    return [(path, label) for path in declared_cache_paths(step)]


def _implicit_owner(step: dict, reference: str) -> list[tuple[str, str]]:
    """Return the paths an action caches without the caller naming them."""
    if not cache_is_enabled(step):
        return []
    if reference.startswith("astral-sh/setup-uv@"):
        return [(UV_CACHE_PATH, "astral-sh/setup-uv")]
    action = shared_action_name(reference)
    if action in SHARED_ACTION_OWNED_PATHS:
        return [(path, action) for path in SHARED_ACTION_OWNED_PATHS[action]]
    return []


def _step_owners(step: dict) -> list[tuple[str, str]]:
    """Return every (path, owner) pair one step establishes."""
    reference = uses_reference(step)
    if CACHE_ACTION_RE.match(reference):
        return _explicit_owner(step, reference)
    return _implicit_owner(step, reference)


def _owned_paths(definition: dict) -> list[tuple[str, str]]:
    """Return every (path, owner) pair a job establishes."""
    return [pair for step in steps(definition) for pair in _step_owners(step)]


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


#: Each installer script and the cache-step path that must restore its work
#: first. Pairing them by tool keeps the ordering check meaningful once a job
#: installs more than one tool.
INSTALLER_CACHE_PAIRS = (("scripts/install-verus.sh", ".verus"),)


def _first_index(items: list[str], predicate: cabc.Callable[[str], bool]) -> int | None:
    """Return the first index whose item satisfies the predicate."""
    return next((index for index, item in enumerate(items) if predicate(item)), None)


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_cache_setup_precedes_the_install_it_protects(workflow_name: str) -> None:
    """A cache restored after an install has already paid for the install."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        job_steps = steps(definition)
        scripts = [run_script(step) for step in job_steps]
        for installer, cached_path in INSTALLER_CACHE_PAIRS:
            install_at = _first_index(scripts, lambda s, i=installer: i in s)
            if install_at is None:
                continue
            cache_at = next(
                (
                    index
                    for index, step in enumerate(job_steps)
                    if CACHE_ACTION_RE.match(uses_reference(step))
                    and any(
                        path.endswith(cached_path)
                        for path in declared_cache_paths(step)
                    )
                ),
                None,
            )
            assert cache_at is not None, (
                f"{workflow_name}:{job_name} runs {installer} with no cache step "
                f"owning {cached_path}"
            )
            assert cache_at < install_at, (
                f"{workflow_name}:{job_name} restores the {cached_path} cache "
                "after the install it is meant to avoid"
            )
