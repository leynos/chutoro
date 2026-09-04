"""Contract-test who owns each cached path.

A mutable cache path with two owners has two keys racing to describe it, so
one of them always restores work the other has already invalidated. Every
path below therefore has exactly one owner per job, and every key is
derived from a correctness input a reader can explain. Compiler output is
excluded outright: sccache owns it, and archiving a ``target`` tree beside
sccache duplicates that ownership and inflates the weekly quota. The caches
that were measured and rejected outright live in ``rejected_caches_test``.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import collections
import collections.abc as cabc
import re
import typing as typ
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


def _cache_steps(definition: dict[str, typ.Any]) -> list[dict[str, typ.Any]]:
    """Return the job's explicit cache steps."""
    return [
        step
        for step in steps(definition)
        if CACHE_ACTION_RE.match(uses_reference(step))
    ]


def _explicit_owner(step: dict[str, typ.Any], reference: str) -> list[tuple[str, str]]:
    """Return the paths a cache step names outright."""
    label = step.get("name", reference)
    return [(path, label) for path in declared_cache_paths(step)]


def _implicit_owner(step: dict[str, typ.Any], reference: str) -> list[tuple[str, str]]:
    """Return the paths an action caches without the caller naming them."""
    if not cache_is_enabled(step):
        return []
    if reference.startswith("astral-sh/setup-uv@"):
        return [(UV_CACHE_PATH, "astral-sh/setup-uv")]
    action = shared_action_name(reference)
    if action in SHARED_ACTION_OWNED_PATHS:
        return [(path, action) for path in SHARED_ACTION_OWNED_PATHS[action]]
    return []


def _step_owners(step: dict[str, typ.Any]) -> list[tuple[str, str]]:
    """Return every (path, owner) pair one step establishes."""
    reference = uses_reference(step)
    if CACHE_ACTION_RE.match(reference):
        return _explicit_owner(step, reference)
    return _implicit_owner(step, reference)


#: The split cache actions. A `restore` and a `save` naming the same path
#: under the same key are two halves of one owner, not two owners: the pair
#: is how a job reads an entry at the start and writes it at the end without
#: the combined action's implicit post-step. Counting them separately would
#: make the single-owner contract reject the very shape it exists to
#: encourage.
SPLIT_CACHE_ACTIONS = ("actions/cache/restore", "actions/cache/save")


#: A (path, key) combination, mapped to the halves that have claimed it.
CacheClaims = dict[tuple[str, str], set[str]]


def _split_cache_half(
    step: dict[str, typ.Any], reference: str
) -> tuple[str, str] | None:
    """Return a restore/save step's (half, key), or None if it is neither.

    The half matters as much as the key. Only one `restore` and one `save`
    make a pair; a second `restore` under the same key is a duplicate owner,
    which is exactly what this module exists to catch.
    """
    half = reference.split("@", 1)[0]
    if half not in SPLIT_CACHE_ACTIONS:
        return None
    with_block = step.get("with")
    key = with_block.get("key", "") if isinstance(with_block, dict) else ""
    return half, key


def _deduplicated_step_owners(
    step: dict[str, typ.Any], claimed: CacheClaims
) -> list[tuple[str, str]]:
    """Return a step's (path, owner) pairs, counting one split pair once.

    `claimed` records which halves have already claimed each (path, key), and
    is updated in place. A path counts as a fresh owner when nothing has
    claimed it yet, or when this same half has already claimed it, which is a
    genuine duplicate. Only the complementary half is absorbed.
    """
    reference = uses_reference(step)
    found = _split_cache_half(step, reference)
    if found is None:
        return _step_owners(step)
    half, key = found
    label = step.get("name", reference)
    fresh: list[str] = []
    for path in declared_cache_paths(step):
        halves = claimed.setdefault((path, key), set())
        if not halves or half in halves:
            fresh.append(path)
        halves.add(half)
    return [(path, label) for path in fresh]


def _owned_paths(definition: dict[str, typ.Any]) -> list[tuple[str, str]]:
    """Return every (path, owner) pair a job establishes."""
    claimed: CacheClaims = {}
    return [
        pair
        for step in steps(definition)
        for pair in _deduplicated_step_owners(step, claimed)
    ]


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


def _cache_half(half: str, name: str, path: str, key: str) -> dict[str, typ.Any]:
    """Build one restore or save step, for the ownership unit tests."""
    return {
        "name": name,
        "uses": f"actions/cache/{half}@{CACHE_ACTION_SHA}",
        "with": {"path": path, "key": key},
    }


@pytest.mark.parametrize(
    ("halves", "expected_owners"),
    [
        pytest.param(("restore", "save"), 1, id="complementary-pair"),
        pytest.param(("save", "restore"), 1, id="pair-in-either-order"),
        pytest.param(("restore", "restore"), 2, id="two-restores"),
        pytest.param(("save", "save"), 2, id="two-saves"),
        pytest.param(("restore", "save", "restore"), 2, id="pair-plus-a-third"),
    ],
)
def test_split_cache_halves_collapse_only_in_complementary_pairs(
    halves: tuple[str, ...], expected_owners: int
) -> None:
    """One restore and one save are one owner; two of a kind are two.

    The first version of this collapsed on (path, key) alone, so a job with
    two `restore` steps naming the same path passed the single-owner
    contract. That is the duplicate the contract exists to catch, so it gets
    a test of its own rather than relying on no workflow happening to do it.
    """
    path = "${{ github.workspace }}/.example"
    definition = {
        "steps": [
            _cache_half(half, f"cache {index}", path, "example-key-v1")
            for index, half in enumerate(halves)
        ]
    }
    owners = [owner for owned, owner in _owned_paths(definition) if owned == path]
    assert len(owners) == expected_owners, (
        f"{halves} should establish {expected_owners} owner(s), found {owners}"
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
