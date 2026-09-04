"""Contract-test that the compiler cache is switched on and pointed somewhere.

Installing sccache is not the same as using it, and using it is not the same
as storing anything. Both failures have already happened here, and both
reported success. First `setup-rust` provided the binary and nothing named it
as the compiler wrapper, so every compilation bypassed it while the logs
showed a working cache with zero compile requests. Then the wrapper was named
and pointed at sccache's GitHub Actions backend, which rejected 273 of
build-test's writes and cost 0.28 s per hit against 0.42 s to compile,
pushing the coverage step into the 600 s nextest timeout.

These tests pin the switching-on: the wrapper is named, the directory is
inside the workspace where a cache step can reach it, the binary comes from
the pinned archive rather than from an action that would overwrite the job's
cache endpoint on its way out, and no job installs a cache that nothing will
ever store. Who reads and who writes that directory is
`sccache_cache_entry_test`'s question.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import typing as typ

import pytest
from sccache_support import (
    EXPECTED_CACHE_DIR,
    EXPECTED_WRAPPER,
    EXPORT_ACTION,
    INSTALLER,
    SETUP_RUST,
    WRAPPER_REQUIRED,
    all_job_ids,
    all_jobs,
    step_index,
    wrapper_job_ids,
    wrapper_jobs,
)
from workflow_support import job, run_script, steps, uses_reference


@pytest.mark.parametrize(("workflow_name", "job_name"), WRAPPER_REQUIRED)
def test_the_compile_heavy_jobs_name_the_wrapper(
    workflow_name: str, job_name: str
) -> None:
    """Without this, sccache is installed and every compilation bypasses it."""
    env = job(workflow_name, job_name).get("env", {})
    assert env.get("RUSTC_WRAPPER") == EXPECTED_WRAPPER, (
        f"{workflow_name}:{job_name} must set RUSTC_WRAPPER to "
        f"{EXPECTED_WRAPPER!r}, or its compilations bypass the cache entirely"
    )


@pytest.mark.parametrize(
    ("workflow_name", "job_name", "definition"),
    wrapper_jobs(),
    ids=wrapper_job_ids(),
)
def test_the_wrapper_names_sccache(
    workflow_name: str, job_name: str, definition: dict[str, typ.Any]
) -> None:
    """Only sccache is a supported wrapper here."""
    wrapper = definition["env"]["RUSTC_WRAPPER"]
    assert wrapper == EXPECTED_WRAPPER, (
        f"{workflow_name}:{job_name} sets RUSTC_WRAPPER to {wrapper!r}, "
        f"expected {EXPECTED_WRAPPER!r}"
    )


@pytest.mark.parametrize(
    ("workflow_name", "job_name", "definition"),
    wrapper_jobs(),
    ids=wrapper_job_ids(),
)
def test_the_cache_directory_is_named_and_shared(
    workflow_name: str, job_name: str, definition: dict[str, typ.Any]
) -> None:
    """Left unset, sccache writes to ~/.cache/sccache, which nothing moves.

    That is the failure that reads as success: the server starts, the
    statistics look plausible, and the directory dies with the runner.
    """
    env = definition["env"]
    assert env.get("SCCACHE_DIR") == EXPECTED_CACHE_DIR, (
        f"{workflow_name}:{job_name} must set SCCACHE_DIR to "
        f"{EXPECTED_CACHE_DIR!r}; the default lives outside the workspace "
        "where no cache step can reach it"
    )


@pytest.mark.parametrize(
    ("workflow_name", "job_name", "definition"),
    all_jobs(),
    ids=all_job_ids(),
)
def test_no_job_enables_the_github_actions_backend(
    workflow_name: str, job_name: str, definition: dict[str, typ.Any]
) -> None:
    """The GHA backend was measured and rejected; it must not return."""
    env = definition.get("env")
    if isinstance(env, dict):
        assert "SCCACHE_GHA_ENABLED" not in env, (
            f"{workflow_name}:{job_name} re-enables sccache's GitHub Actions "
            "backend, which rejected 273 writes and cost more per hit than "
            "compiling. See the developers guide."
        )
    for step in steps(definition):
        step_env = step.get("env")
        if isinstance(step_env, dict):
            assert "SCCACHE_GHA_ENABLED" not in step_env, (
                f"{workflow_name}:{job_name} re-enables sccache's GitHub "
                "Actions backend in a step"
            )
        assert "SCCACHE_GHA_ENABLED" not in run_script(step), (
            f"{workflow_name}:{job_name} exports SCCACHE_GHA_ENABLED from a "
            "script; the GitHub Actions backend was measured and rejected"
        )


@pytest.mark.parametrize(
    ("workflow_name", "job_name", "definition"),
    all_jobs(),
    ids=all_job_ids(),
)
def test_no_job_installs_a_cache_it_cannot_store(
    workflow_name: str, job_name: str, definition: dict[str, typ.Any]
) -> None:
    """A cache with no store is worse than none: it reports success anyway.

    `setup-rust` would install sccache through an action whose last act
    overwrites the job's cache endpoint in `GITHUB_ENV`. Every job here
    installs the pinned binary from a `run:` step instead, and jobs with no
    cache entry to read do not install it at all.
    """
    for step in steps(definition):
        if SETUP_RUST not in uses_reference(step):
            continue
        with_block = step.get("with")
        enabled = (
            with_block.get("use-sccache", "true")
            if isinstance(with_block, dict)
            else "true"
        )
        assert str(enabled) == "false", (
            f"{workflow_name}:{job_name} lets Setup Rust install sccache; "
            "that action rewrites the job's cache endpoint in GITHUB_ENV on "
            "its way out. Pass use-sccache: 'false'."
        )


@pytest.mark.parametrize(
    ("workflow_name", "job_name", "definition"),
    wrapper_jobs(),
    ids=wrapper_job_ids(),
)
def test_the_binary_is_installed_from_the_pinned_archive(
    workflow_name: str, job_name: str, definition: dict[str, typ.Any]
) -> None:
    """Nothing is built from source in CI, and nothing floats."""
    assert step_index(definition, lambda s: INSTALLER in run_script(s)) is not None, (
        f"{workflow_name}:{job_name} names sccache as its wrapper but never "
        f"runs {INSTALLER}, so the binary is unpinned or absent"
    )


@pytest.mark.parametrize(
    ("workflow_name", "job_name", "definition"),
    all_jobs(),
    ids=all_job_ids(),
)
def test_no_job_exports_the_ubicloud_cache_credentials(
    workflow_name: str, job_name: str, definition: dict[str, typ.Any]
) -> None:
    """Nothing needs them once the GitHub Actions backend is gone.

    The export republishes the runner's cache-proxy URL and token so a
    `run:`-started sccache server can reach the proxy's v1 cache service.
    Only the abandoned backend spoke that protocol. `actions/cache` is an
    action step, so the runner hands it those variables directly, and an
    export left behind would be a live credential in the job environment
    serving nothing.
    """
    exporters = [
        index
        for index, step in enumerate(steps(definition))
        if EXPORT_ACTION in uses_reference(step)
    ]
    assert not exporters, (
        f"{workflow_name}:{job_name} exports Ubicloud cache credentials at "
        f"steps {exporters}; no job uses sccache's GitHub Actions backend, "
        "and actions/cache reaches the proxy without them"
    )
