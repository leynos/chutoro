"""Shared vocabulary for the compiler-cache contracts.

The wiring contracts and the cache-entry contracts ask different questions
of the same workflows, and both need the same handful of names: which
wrapper, which directory, which installer, which job may write. Restating
them in each module is how two contracts come to disagree about what the
arrangement is.

The measurements behind those choices are in the developers guide, under
"The compiler cache".
"""

from __future__ import annotations

import re
import typing as typ

from workflow_support import (
    declared_cache_paths,
    jobs,
    load_workflow,
    run_script,
    steps,
    uses_reference,
    workflow_names,
)

#: The wrapper every instrumented job must name. The installer puts the
#: binary in ~/.cargo/bin, which is on PATH, so the bare name resolves.
EXPECTED_WRAPPER = "sccache"

#: The directory holding the cache. It sits inside the workspace so
#: `actions/cache`'s relative path resolves and .gitignore keeps it out of
#: the tree coverage and CodeScene read.
EXPECTED_CACHE_DIR = "${{ github.workspace }}/.sccache"

#: The installer that resolves the pinned, checksum-verified archive.
INSTALLER = "scripts/install-sccache.sh"

#: The jobs that must name the wrapper. This is a list rather than a sweep
#: because a sweep over "jobs that set RUSTC_WRAPPER" quietly shrinks to
#: nothing when someone removes the variable, which is the exact regression
#: worth catching. Only the two jobs sharing the build-test key are here:
#: they are the pair with a reader and a writer. The property suites,
#: benchmark jobs and nightly jobs deliberately carry no compiler cache,
#: which the coverage of `test_no_job_installs_a_cache_it_cannot_store`
#: keeps honest.
WRAPPER_REQUIRED = (
    ("ci.yml", "build-test"),
    ("coverage-main.yml", "coverage-upload"),
)

#: The one job allowed to write the compiler-cache key, and the event that
#: may trigger the write. A dispatch restores and never saves, so a manual
#: re-run cannot overwrite the entry a real merge produced.
EXPECTED_WRITER = ("coverage-main.yml", "coverage-upload")
EXPECTED_WRITE_EVENT = "github.event_name == 'push'"

#: A step that actually compiles something. The statistics are meaningless
#: unless one of these runs between the reset and the report. Compilation
#: also happens inside shared actions, so those count as build steps too.
BUILD_COMMAND = re.compile(r"\bcargo\s+(nextest|test|build|clippy|llvm-cov)\b")
BUILD_ACTIONS = ("/actions/generate-coverage@", "/actions/rust-build-release@")

#: The shared action that republishes Ubicloud's cache-proxy credentials.
#: It exists to let a `run:`-started sccache server reach the proxy's v1
#: cache service, which only the abandoned GitHub Actions backend needed.
#: `actions/cache` reaches the proxy natively from an action step, so no
#: job here should be exporting anything.
EXPORT_ACTION = (
    "leynos/shared-actions/.github/actions/export-ubicloud-cache-credentials@"
)

#: The shared action whose sccache install must stay switched off. It runs
#: mozilla-actions/sccache-action, whose last act is to write
#: ACTIONS_CACHE_SERVICE_V2 and GitHub's results URL and token to GITHUB_ENV,
#: clobbering any earlier cache-endpoint export for the rest of the job.
SETUP_RUST = "/actions/setup-rust@"


def is_build_step(step: dict[str, typ.Any]) -> bool:
    """Report whether a step compiles Rust."""
    if BUILD_COMMAND.search(run_script(step)):
        return True
    reference = uses_reference(step)
    return any(action in reference for action in BUILD_ACTIONS)


def is_sccache_cache_step(step: dict[str, typ.Any]) -> bool:
    """Report whether a step restores or saves the compiler-cache directory."""
    if not uses_reference(step).startswith("actions/cache"):
        return False
    return EXPECTED_CACHE_DIR in declared_cache_paths(step)


def all_jobs() -> list[tuple[str, str, dict[str, typ.Any]]]:
    """Return every job in every workflow."""
    return [
        (workflow_name, job_name, definition)
        for workflow_name in workflow_names()
        for job_name, definition in jobs(load_workflow(workflow_name)).items()
    ]


def all_job_ids() -> list[str]:
    """Return stable identifiers for the whole-estate parametrization."""
    return [f"{workflow}:{name}" for workflow, name, _ in all_jobs()]


def wrapper_jobs() -> list[tuple[str, str, dict[str, typ.Any]]]:
    """Return every job that names a compiler wrapper."""
    return [
        (workflow_name, job_name, definition)
        for workflow_name, job_name, definition in all_jobs()
        if isinstance(definition.get("env"), dict)
        and "RUSTC_WRAPPER" in definition["env"]
    ]


def wrapper_job_ids() -> list[str]:
    """Return stable identifiers for the wrapper-job parametrization."""
    return [f"{workflow}:{name}" for workflow, name, _ in wrapper_jobs()]


def step_index(
    definition: dict[str, typ.Any],
    predicate: typ.Callable[[dict[str, typ.Any]], bool],
) -> int | None:
    """Return the index of the first step satisfying a predicate."""
    return next(
        (index for index, step in enumerate(steps(definition)) if predicate(step)),
        None,
    )
