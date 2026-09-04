"""Contract-test that the compiler cache is actually wired up.

Installing sccache is not the same as using it, and using it is not the
same as storing anything. Both failures have already happened here. First
`setup-rust` provided the binary and nothing named it as the compiler
wrapper, so every compilation bypassed it while the logs reported a
working cache with zero compile requests. Then the wrapper was named and
pointed at sccache's GitHub Actions backend, which rejected 273 of
build-test's writes and cost 0.28 s per hit against 0.42 s to compile,
pushing the coverage step into the 600 s nextest timeout.

What survives both is the local-disk arm: a pinned sccache binary, a
directory under the workspace, and `actions/cache` moving that directory.
These tests pin the parts that make it real. The wrapper is named. The
binary comes from the pinned archive, not from an action that would start
the server where the runner re-injects the reserved cache variables. The
directory is restored before the server starts. The counters are zeroed
before the build so the statistics describe this run. The statistics reach
the log, not only the job summary. And exactly one job writes the key, on
push to main, after reclaiming the disk it needs to build the archive.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import re
import typing as typ

import pytest
from workflow_support import (
    CACHE_ACTION_SHA,
    declared_cache_paths,
    job,
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
#: mozilla-actions/sccache-action, which starts the server inside an action
#: step; the runner re-injects the reserved cache variables there, so such a
#: server binds GitHub's cache service whatever the job asked for.
SETUP_RUST = "/actions/setup-rust@"


def _is_build_step(step: dict[str, typ.Any]) -> bool:
    """Report whether a step compiles Rust."""
    if BUILD_COMMAND.search(run_script(step)):
        return True
    reference = uses_reference(step)
    return any(action in reference for action in BUILD_ACTIONS)


def _is_sccache_cache_step(step: dict[str, typ.Any]) -> bool:
    """Report whether a step restores or saves the compiler-cache directory."""
    if not uses_reference(step).startswith("actions/cache"):
        return False
    return EXPECTED_CACHE_DIR in declared_cache_paths(step)


def _all_jobs() -> list[tuple[str, str, dict[str, typ.Any]]]:
    """Return every job in every workflow."""
    return [
        (workflow_name, job_name, definition)
        for workflow_name in workflow_names()
        for job_name, definition in jobs(load_workflow(workflow_name)).items()
    ]


def _all_job_ids() -> list[str]:
    """Return stable identifiers for the whole-estate parametrization."""
    return [f"{workflow}:{name}" for workflow, name, _ in _all_jobs()]


def _wrapper_jobs() -> list[tuple[str, str, dict[str, typ.Any]]]:
    """Return every job that names a compiler wrapper."""
    return [
        (workflow_name, job_name, definition)
        for workflow_name, job_name, definition in _all_jobs()
        if isinstance(definition.get("env"), dict)
        and "RUSTC_WRAPPER" in definition["env"]
    ]


def _wrapper_job_ids() -> list[str]:
    """Return stable identifiers for the wrapper-job parametrization."""
    return [f"{workflow}:{name}" for workflow, name, _ in _wrapper_jobs()]


def _step_index(
    definition: dict[str, typ.Any],
    predicate: typ.Callable[[dict[str, typ.Any]], bool],
) -> int | None:
    """Return the index of the first step satisfying a predicate."""
    return next(
        (index for index, step in enumerate(steps(definition)) if predicate(step)),
        None,
    )


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
    _wrapper_jobs(),
    ids=_wrapper_job_ids(),
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
    _wrapper_jobs(),
    ids=_wrapper_job_ids(),
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
    _all_jobs(),
    ids=_all_job_ids(),
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
    _all_jobs(),
    ids=_all_job_ids(),
)
def test_no_job_installs_a_cache_it_cannot_store(
    workflow_name: str, job_name: str, definition: dict[str, typ.Any]
) -> None:
    """A cache with no store is worse than none: it reports success anyway.

    `setup-rust` would install sccache and start its server inside an
    action step, where the runner re-injects the reserved cache variables.
    Every job here installs the pinned binary from a `run:` step instead,
    and jobs with no cache entry to read do not install it at all.
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
            f"{workflow_name}:{job_name} lets Setup Rust install and start "
            "sccache; that server binds GitHub's cache service regardless of "
            "the job's settings. Pass use-sccache: 'false'."
        )


@pytest.mark.parametrize(
    ("workflow_name", "job_name", "definition"),
    _wrapper_jobs(),
    ids=_wrapper_job_ids(),
)
def test_the_binary_is_installed_from_the_pinned_archive(
    workflow_name: str, job_name: str, definition: dict[str, typ.Any]
) -> None:
    """Nothing is built from source in CI, and nothing floats."""
    assert _step_index(definition, lambda s: INSTALLER in run_script(s)) is not None, (
        f"{workflow_name}:{job_name} names sccache as its wrapper but never "
        f"runs {INSTALLER}, so the binary is unpinned or absent"
    )


@pytest.mark.parametrize(
    ("workflow_name", "job_name", "definition"),
    _wrapper_jobs(),
    ids=_wrapper_job_ids(),
)
def test_the_cache_is_restored_before_the_server_starts(
    workflow_name: str, job_name: str, definition: dict[str, typ.Any]
) -> None:
    """Order decides whether the restored directory is ever read.

    The server reads SCCACHE_DIR when it starts. Unpacking the archive
    underneath a running server leaves the run compiling from scratch while
    the statistics report a full cache.
    """
    install_at = _step_index(definition, lambda s: INSTALLER in run_script(s))
    restore_at = _step_index(definition, _is_sccache_cache_step)
    start_at = _step_index(definition, lambda s: "--zero-stats" in run_script(s))
    assert restore_at is not None, (
        f"{workflow_name}:{job_name} names sccache as its wrapper with no "
        "cache step for its directory, so nothing survives the runner"
    )
    assert install_at is not None and start_at is not None
    assert install_at < restore_at < start_at, (
        f"{workflow_name}:{job_name} must install, then restore, then start "
        f"the server; found install={install_at}, restore={restore_at}, "
        f"start={start_at}"
    )


@pytest.mark.parametrize(
    ("workflow_name", "job_name", "definition"),
    _wrapper_jobs(),
    ids=_wrapper_job_ids(),
)
def test_every_compiler_cache_step_explains_its_key(
    workflow_name: str, job_name: str, definition: dict[str, typ.Any]
) -> None:
    """A key that omits a correctness input restores work it cannot reuse.

    The lockfile and the toolchain both change every hash sccache computes,
    so both belong in the key. Readers additionally need a prefix in
    restore-keys, or a lockfile bump drops them to a cold start rather than
    to yesterday's entry.
    """
    cache_steps = [step for step in steps(definition) if _is_sccache_cache_step(step)]
    assert cache_steps, f"{workflow_name}:{job_name} has no compiler-cache step"
    for step in cache_steps:
        with_block = step.get("with", {})
        key = with_block.get("key", "")
        assert CACHE_ACTION_SHA in uses_reference(step), (
            f"{workflow_name}:{job_name} pins the compiler cache at a version "
            f"other than {CACHE_ACTION_SHA}"
        )
        for component in ("Cargo.lock", "rust-toolchain.toml"):
            assert component in key, (
                f"{workflow_name}:{job_name} keys the compiler cache without "
                f"{component}; a change to it would restore unusable entries"
            )
        if uses_reference(step).startswith("actions/cache/save"):
            continue
        restore_keys = with_block.get("restore-keys", "")
        assert restore_keys.strip(), (
            f"{workflow_name}:{job_name} restores the compiler cache with no "
            "restore-keys prefix, so any key change is a cold start"
        )


def test_exactly_one_job_writes_the_compiler_cache() -> None:
    """Two writers race on every merge and the loser's work is discarded."""
    writers = [
        (workflow_name, job_name, step)
        for workflow_name, job_name, definition in _all_jobs()
        for step in steps(definition)
        if uses_reference(step).startswith("actions/cache/save")
        and EXPECTED_CACHE_DIR in declared_cache_paths(step)
    ]
    located = [(workflow_name, job_name) for workflow_name, job_name, _ in writers]
    assert located == [EXPECTED_WRITER], (
        f"the compiler cache must be written by {EXPECTED_WRITER} alone; "
        f"found {located}"
    )
    condition = writers[0][2].get("if", "")
    assert EXPECTED_WRITE_EVENT in condition, (
        "the compiler-cache save must be guarded by "
        f"{EXPECTED_WRITE_EVENT!r}; a dispatch restores and never saves, or a "
        "manual re-run overwrites the entry a merge produced"
    )


def test_the_writer_reclaims_disk_before_saving() -> None:
    """The archive is built on the same disk the coverage tree fills.

    `ubicloud-standard-2` starts with about 31 GB free and the coverage
    scratch tree is the largest thing on it, with no consumer after the
    report. Deleting it before the save is what leaves room to build the
    archive; `df -h` on both sides is how the next reader knows it did.
    """
    definition = job(*EXPECTED_WRITER)
    reclaim_at = _step_index(
        definition,
        lambda s: "df -h" in run_script(s) and "rm -rf target/" in run_script(s),
    )
    save_at = _step_index(
        definition, lambda s: uses_reference(s).startswith("actions/cache/save")
    )
    build_at = _step_index(definition, _is_build_step)
    assert reclaim_at is not None, (
        f"{EXPECTED_WRITER[0]}:{EXPECTED_WRITER[1]} must delete its scratch "
        "trees and print df -h before building the cache archive"
    )
    assert save_at is not None and build_at is not None
    assert build_at < reclaim_at < save_at, (
        "the reclaim must sit between the build and the save; found "
        f"build={build_at}, reclaim={reclaim_at}, save={save_at}"
    )


@pytest.mark.parametrize(
    ("workflow_name", "job_name", "definition"),
    _wrapper_jobs(),
    ids=_wrapper_job_ids(),
)
def test_counters_are_zeroed_then_reported(
    workflow_name: str, job_name: str, definition: dict[str, typ.Any]
) -> None:
    """Statistics only mean something when they describe one build."""
    scripts = [run_script(step) for step in steps(definition)]
    zero_at = next(
        (index for index, script in enumerate(scripts) if "--zero-stats" in script),
        None,
    )
    show_at = next(
        (index for index, script in enumerate(scripts) if "--show-stats" in script),
        None,
    )
    assert zero_at is not None, (
        f"{workflow_name}:{job_name} uses the compiler cache without zeroing "
        "its counters, so its statistics describe earlier work too"
    )
    assert show_at is not None, (
        f"{workflow_name}:{job_name} uses the compiler cache without reporting "
        "its statistics, so nobody can tell whether it worked"
    )
    assert zero_at < show_at, (
        f"{workflow_name}:{job_name} zeroes its counters after reporting them"
    )
    # The build has to sit between the two. Zeroing after compilation and
    # before reporting satisfies the ordering above while reporting nothing
    # but zeros, which looks like a working cache that did no work.
    build_at = next(
        (
            index
            for index, step in enumerate(steps(definition))
            if index > zero_at and _is_build_step(step)
        ),
        None,
    )
    assert build_at is not None, (
        f"{workflow_name}:{job_name} zeroes its counters but never compiles "
        "afterwards, so the statistics describe no work"
    )
    assert build_at < show_at, (
        f"{workflow_name}:{job_name} reports its statistics before the build "
        "they are meant to describe"
    )
    report = scripts[show_at]
    assert "GITHUB_STEP_SUMMARY" in report, (
        f"{workflow_name}:{job_name} must write its statistics to the job "
        "summary, where an operator can read them without opening the log"
    )
    # Job summaries are not exposed through the REST API, so statistics that
    # only reach the summary cannot be checked by anything except a human
    # with a browser. `tee` puts them in both places.
    assert "tee" in report, (
        f"{workflow_name}:{job_name} must also emit its statistics to the "
        "log; a summary-only report is unreadable by tooling"
    )
    # Which backend sccache bound to decides whether any other number in the
    # report means anything, and it is not in `--show-stats` output until the
    # `Cache location` line, which is easy to miss. Name it explicitly.
    assert "ACTIONS_CACHE_SERVICE_V2" in report, (
        f"{workflow_name}:{job_name} must report which cache service was "
        "selected; without it the statistics cannot be interpreted"
    )
    # A restore that silently missed produces exactly the numbers a cold run
    # produces, and no reader can distinguish the two without knowing which
    # key answered.
    assert "cache-matched-key" in str(steps(definition)[show_at].get("env", {})), (
        f"{workflow_name}:{job_name} must report which cache key its restore "
        "matched; without it a silent miss reads as a cold start"
    )


@pytest.mark.parametrize(
    ("workflow_name", "job_name", "definition"),
    _all_jobs(),
    ids=_all_job_ids(),
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
