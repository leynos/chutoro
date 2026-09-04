"""Contract-test that the caches measured and rejected stay rejected.

A cache that costs more than it saves is easy to add and hard to notice, so
the two that failed the payoff rule are named here rather than left to
memory. Kani's three parts total about 1.8 GB against a 16 s cold install
(run 33819842254); cargo-nextest is an 11 MB archive to save downloading an
11 MB archive. Both still install from pinned, checksum-verified releases.

Path matching goes through `cache_path_globs`, because exact equality is not
enough in three separate ways: trailing separators, parents sweeping their
children in, and glob patterns that name neither.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import typing as typ

import pytest
from cache_path_globs import covers
from workflow_support import (
    declared_cache_paths,
    jobs,
    load_workflow,
    steps,
    uses_reference,
    workflow_names,
)

CACHE_ACTION_PREFIX = "actions/cache"


def _cache_steps(definition: dict[str, typ.Any]) -> list[dict[str, typ.Any]]:
    """Return the job's explicit cache steps."""
    return [
        step
        for step in steps(definition)
        if uses_reference(step).split("@", 1)[0].startswith(CACHE_ACTION_PREFIX)
    ]


#: Paths that must never reappear in a cache step. Kani's three parts total
#: about 1.8 GB against a 16 s cold install (run 33819842254), so caching them
#: costs far more runner time than it saves. Removing the old contract without
#: this one would let the archive come back unnoticed.
REJECTED_CACHE_PATHS: tuple[str, ...] = (
    "~/.cargo/bin/cargo-kani",
    "~/.cargo/bin/kani",
    "~/.kani",
    "~/.kani-rustup",
)


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_the_rejected_caches_stay_rejected(workflow_name: str) -> None:
    """A cache that failed the payoff rule must not quietly return."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        for step in _cache_steps(definition):
            for path in declared_cache_paths(step):
                covered = [
                    rejected
                    for rejected in REJECTED_CACHE_PATHS
                    if covers(path, rejected)
                ]
                assert not covered, (
                    f"{workflow_name}:{job_name} caches {path}, which covers "
                    f"{covered}; those were measured and rejected. See the "
                    "developers guide."
                )


@pytest.mark.parametrize(
    ("declared", "is_covered"),
    [
        pytest.param("~/.kani", True, id="exact"),
        pytest.param("~/.kani/", True, id="trailing-separator"),
        pytest.param("~/.cargo/bin", True, id="parent-directory"),
        pytest.param("~/.kani-rustup/toolchains", True, id="child-directory"),
        pytest.param("~/.cargo/registry", False, id="unrelated-sibling"),
        pytest.param("~/.cargo/bin/whitaker-installer", False, id="other-tool"),
        pytest.param("~/.kani-extra", False, id="shared-prefix-only"),
        pytest.param("~/.cargo/bin/*", True, id="globbed-parent"),
        pytest.param("~/.cargo/**", True, id="recursive-glob"),
        pytest.param("~/.kani*", True, id="glob-matching-the-path-itself"),
        pytest.param("~/.kani-rustup/*/lib", True, id="glob-inside-a-rejected-tree"),
        pytest.param("~/.cargo/bin/whitaker-*", False, id="glob-for-another-tool"),
        pytest.param("~/.cargo/registry/**", False, id="recursive-glob-elsewhere"),
        pytest.param("~/.cargo/bin/?ani", True, id="single-character-wildcard"),
        pytest.param("~/.cargo/bin/[ck]ani", True, id="bracket-expression"),
    ],
)
def test_rejected_path_coverage_recognizes_equivalent_spellings(
    declared: str, is_covered: bool
) -> None:
    """Guard the guard, since exact equality was the original hole."""
    covered = any(covers(declared, rejected) for rejected in REJECTED_CACHE_PATHS)
    assert covered is is_covered, (
        f"{declared} should {'' if is_covered else 'not '}count as covering a "
        "rejected cache path"
    )
