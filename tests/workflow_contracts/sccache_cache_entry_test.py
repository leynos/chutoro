"""Contract-test who reads and who writes the compiler cache, and in what order.

A directory sccache writes to is worth nothing unless something moves it off
the runner, and worth less than nothing if two jobs move it under competing
keys. `ci.yml`'s build-test restores the key on every pull request and never
saves; `coverage-main.yml`'s coverage-upload compiles the same workspace under
the same instrumentation on push to main and is the only job that saves.

Order is as load-bearing as ownership. The server reads `SCCACHE_DIR` when it
starts, so unpacking the archive underneath a running server leaves the run
compiling from scratch while the statistics report a full cache. The counters
have to be zeroed before the build and read after it, and the restored key has
to be reported, because a restore that silently missed produces exactly the
numbers a cold run produces.

Whether the cache is switched on at all is `sccache_wiring_test`'s question.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import re
import typing as typ

import pytest
from sccache_support import (
    BUILD_ACTIONS,
    EXPECTED_CACHE_DIR,
    EXPECTED_WRITE_EVENT,
    EXPECTED_WRITER,
    INSTALLER,
    all_jobs,
    is_build_step,
    is_sccache_cache_step,
    step_index,
    wrapper_job_ids,
    wrapper_jobs,
)
from workflow_support import (
    CACHE_ACTION_SHA,
    declared_cache_paths,
    job,
    run_script,
    steps,
    uses_reference,
)


@pytest.mark.parametrize(
    ("workflow_name", "job_name", "definition"),
    wrapper_jobs(),
    ids=wrapper_job_ids(),
)
def test_the_cache_is_restored_before_the_server_starts(
    workflow_name: str, job_name: str, definition: dict[str, typ.Any]
) -> None:
    """Order decides whether the restored directory is ever read.

    The server reads SCCACHE_DIR when it starts. Unpacking the archive
    underneath a running server leaves the run compiling from scratch while
    the statistics report a full cache.
    """
    install_at = step_index(definition, lambda s: INSTALLER in run_script(s))
    restore_at = step_index(definition, is_sccache_cache_step)
    start_at = step_index(definition, lambda s: "--zero-stats" in run_script(s))
    assert restore_at is not None, (
        f"{workflow_name}:{job_name} names sccache as its wrapper with no "
        "cache step for its directory, so nothing survives the runner"
    )
    assert install_at is not None, (
        f"{workflow_name}:{job_name} names sccache as its wrapper but never "
        f"runs {INSTALLER}, so there is no server to restore into"
    )
    assert start_at is not None, (
        f"{workflow_name}:{job_name} restores the cache but never starts the "
        "server with --zero-stats, so the directory is never read"
    )
    assert install_at < restore_at < start_at, (
        f"{workflow_name}:{job_name} must install, then restore, then start "
        f"the server; found install={install_at}, restore={restore_at}, "
        f"start={start_at}"
    )


@pytest.mark.parametrize(
    ("workflow_name", "job_name", "definition"),
    wrapper_jobs(),
    ids=wrapper_job_ids(),
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
    cache_steps = [step for step in steps(definition) if is_sccache_cache_step(step)]
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
        for workflow_name, job_name, definition in all_jobs()
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


#: The job that reads the compiler-cache key. Its compile steps are the
#: shapes the writer has to produce.
EXPECTED_READER: tuple[str, str] = ("ci.yml", "build-test")

#: Inputs that change what a shared action compiles, per action. Only these
#: belong in a shape. `output-path` and `format` decide where the report
#: goes; `use-cargo-nextest` decides whether the workspace is built for
#: nextest or for `cargo test`, which is a different set of objects, and
#: `with-ratchet` adds a baseline comparison build. An action input absent
#: from this mapping is deliberately ignored, so a reader and a writer may
#: differ on where they write their coverage file without failing the
#: contract.
COMPILATION_RELEVANT_INPUTS: dict[str, tuple[str, ...]] = {
    "generate-coverage": ("use-cargo-nextest", "with-ratchet"),
    "rust-build-release": ("target", "features", "profile"),
}

#: Cargo subcommands that produce compiler output. `fmt` and `metadata` do
#: not, so they are not shapes the cache has to hold.
COMPILING_CARGO: re.Pattern[str] = re.compile(
    r"^cargo\s+(?:\+\S+\s+)?(?:nextest|test|build|clippy|check|doc|bench|run|llvm-cov)\b"
)

#: Make targets that compile. `check-fmt`, `spelling` and
#: `test-workflow-contracts` do not, so a writer need not run them; `lint`
#: does, through rustdoc, Clippy and the Whitaker Dylint suite, and it is
#: the shape whose absence left the warm hit rate stuck.
#: The trailing guard is load-bearing: a `\b` after `test` would also match
#: `make test-workflow-contracts`, which runs pytest and compiles nothing.
COMPILING_MAKE: re.Pattern[str] = re.compile(
    r"^make\s+(?:lint|lint-clippy|lint-whitaker|test|typecheck|build|release|bench)"
    r"(?:\s|$)"
)


# A shell line continuation is a formatting choice, not a different command,
# so a `cargo clippy` split across four lines has to read as the single shape
# it is. Whitespace is collapsed for the same reason: a reflow must not look
# like a new command to the contract.
def _command_lines(script: str) -> list[str]:
    """Return a script's commands, one per line, with continuations joined."""
    joined = script.replace("\\\n", " ")
    return [" ".join(line.split()) for line in joined.splitlines() if line.strip()]


def _script_shapes(step: dict[str, typ.Any]) -> set[str]:
    """Return the compiling commands a step's `run:` script invokes."""
    return {
        line
        for line in _command_lines(run_script(step))
        if COMPILING_CARGO.match(line) or COMPILING_MAKE.match(line)
    }


# The action's reference stands in for the command, because the caller cannot
# see what it runs. The pinned revision is part of the shape: two jobs on
# different revisions of `generate-coverage` may compile differently, and
# reducing both to the bare path would hide exactly that drift. So are the
# inputs that change what is built, which is why they are named explicitly
# rather than folded in wholesale; see COMPILATION_RELEVANT_INPUTS.
def _action_shape(reference: str, with_block: dict[str, typ.Any]) -> str:
    """Return the canonical shape for one compiling action reference."""
    path, _, revision = reference.partition("@")
    action = path.rsplit("/", 1)[-1]
    relevant = COMPILATION_RELEVANT_INPUTS.get(action, ())
    inputs = " ".join(
        f"{name}={with_block[name]}" for name in relevant if name in with_block
    )
    canonical = f"{path}@{revision}" if revision else path
    return f"{canonical} {inputs}".rstrip()


def _action_shapes(step: dict[str, typ.Any]) -> set[str]:
    """Return the compiling shared action a step uses, if it uses one."""
    reference = uses_reference(step)
    if not any(action in reference for action in BUILD_ACTIONS):
        return set()
    with_block = step.get("with")
    return {
        _action_shape(reference, with_block if isinstance(with_block, dict) else {})
    }


# A shape is a whole command, not just its subcommand. `cargo clippy -p
# chutoro-providers-dense --no-default-features --features simd_avx2` produces
# different objects from a plain workspace Clippy run, and sccache keys them
# separately, so a writer that runs only the latter leaves the former missing
# forever.
def _compile_shapes(definition: dict[str, typ.Any]) -> set[str]:
    """Return every distinct compilation a job performs."""
    return {
        shape
        for step in steps(definition)
        for shape in _script_shapes(step) | _action_shapes(step)
    }


def test_the_writer_compiles_every_shape_the_reader_reads() -> None:
    """A writer that builds less than its reader leaves permanent misses.

    One key may have exactly one owner, so a shape the writer never builds
    is a shape no pull request can ever hit. That is not a cache fault and
    it does not heal: it measured as a warm hit rate stuck at 47.94 % with
    byte-identical counters across two dispatches, 1375 hits and 1493
    misses each time. A flaky cache varies; a structurally incomplete
    archive does not.

    The two jobs live in different files, so YAML anchors cannot hold them
    together. This does.
    """
    reader = _compile_shapes(job(*EXPECTED_READER))
    writer = _compile_shapes(job(*EXPECTED_WRITER))
    missing = sorted(reader - writer)
    assert not missing, (
        f"{EXPECTED_WRITER[0]}:{EXPECTED_WRITER[1]} writes the compiler-cache "
        f"key that {EXPECTED_READER[0]}:{EXPECTED_READER[1]} reads, but never "
        f"compiles {missing}. Every object those commands produce would miss "
        "on every pull request, permanently."
    )


def test_the_writer_reclaims_disk_before_saving() -> None:
    """The archive is built on the same disk the coverage tree fills.

    `ubicloud-standard-2` starts with about 31 GB free and the coverage
    scratch tree is the largest thing on it, with no consumer after the
    report. Deleting it before the save is what leaves room to build the
    archive; `df -h` on both sides is how the next reader knows it did.
    """
    definition = job(*EXPECTED_WRITER)
    reclaim_at = step_index(
        definition,
        lambda s: "df -h" in run_script(s) and "rm -rf target/" in run_script(s),
    )
    save_at = step_index(
        definition, lambda s: uses_reference(s).startswith("actions/cache/save")
    )
    build_at = step_index(definition, is_build_step)
    assert reclaim_at is not None, (
        f"{EXPECTED_WRITER[0]}:{EXPECTED_WRITER[1]} must delete its scratch "
        "trees and print df -h before building the cache archive"
    )
    assert save_at is not None, (
        f"{EXPECTED_WRITER[0]}:{EXPECTED_WRITER[1]} is the designated writer "
        "but declares no cache save step"
    )
    assert build_at is not None, (
        f"{EXPECTED_WRITER[0]}:{EXPECTED_WRITER[1]} saves a compiler cache "
        "without compiling anything, so the archive would hold nothing new"
    )
    assert build_at < reclaim_at < save_at, (
        "the reclaim must sit between the build and the save; found "
        f"build={build_at}, reclaim={reclaim_at}, save={save_at}"
    )


@pytest.mark.parametrize(
    ("workflow_name", "job_name", "definition"),
    wrapper_jobs(),
    ids=wrapper_job_ids(),
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
            if index > zero_at and is_build_step(step)
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
