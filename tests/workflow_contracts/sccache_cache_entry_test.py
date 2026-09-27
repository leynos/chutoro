"""Contract-test that `main` writes every shape a pull request compiles.

Ubicloud's cache proxy is ref-scoped: a pull request reads `main`'s scope and
its own, and nothing else. `setup-rust` writes every compilation to the
running ref's scope, so a shape that no push to `main` compiles is a shape a
pull request's first push can never hit. `ci.yml`'s `build-test` runs only
on pull requests; `coverage-main.yml`'s `coverage-upload` is the job that
compiles on every push to `main`, so its compile commands must cover
`build-test`'s. When it built only the coverage shape, the warm hit rate
stalled at 47.94 % with byte-identical counters across two dispatches; the
same gap would reopen on the proxy.

The statistics have to be reported after the build and name the backend
`setup-rust` selected, because `Cache location` alone reads `ghac` for the
proxy and GitHub's own service alike.

Whether `setup-rust` owns the cache at all is `sccache_wiring_test`'s
question.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import re
import typing as typ

import pytest
from sccache_support import (
    BUILD_ACTIONS,
    CACHE_JOBS,
    EXPECTED_READER,
    EXPECTED_WRITER,
    SETUP_RUST_ID,
    is_build_step,
)
from workflow_support import job, run_script, steps, uses_reference

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
    inputs = with_block if isinstance(with_block, dict) else {}
    return {_action_shape(reference, inputs)}


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
    """A trunk that builds less than a pull request leaves permanent misses.

    A shape no push to `main` compiles is a shape a pull request's first push
    can never hit. That is not a cache fault and it does not heal: under the
    retired archive it measured as a warm hit rate stuck at 47.94 % with
    byte-identical counters across two dispatches, 1375 hits and 1493 misses
    each time. A flaky cache varies; a structurally incomplete one does not.

    The two jobs live in different files, so YAML anchors cannot hold them
    together. This does.
    """
    reader = _compile_shapes(job(*EXPECTED_READER))
    writer = _compile_shapes(job(*EXPECTED_WRITER))
    missing = sorted(reader - writer)
    assert not missing, (
        f"{EXPECTED_WRITER[0]}:{EXPECTED_WRITER[1]} writes `main`'s cache "
        f"scope, which {EXPECTED_READER[0]}:{EXPECTED_READER[1]} reads, but "
        f"never compiles {missing}. Every object those commands produce would "
        "miss on every pull request's first push, permanently."
    )


@pytest.mark.parametrize(("workflow_name", "job_name"), list(CACHE_JOBS))
def test_statistics_are_reported_after_the_build_with_the_backend(
    workflow_name: str, job_name: str
) -> None:
    """Statistics only mean something when they follow the build they describe.

    `setup-rust` zeroes the counters when it starts the server, so the report
    has to come after the last compiling step, in the log as well as the job
    summary, and name the backend the action chose. Job summaries are not
    exposed through the REST API, so a summary-only report is unreadable by
    tooling, and `Cache location` reads `ghac` whether sccache reached the
    proxy or went past it.
    """
    definition = job(workflow_name, job_name)
    scripts = [run_script(step) for step in steps(definition)]
    show_at = next(
        (index for index, script in enumerate(scripts) if "--show-stats" in script),
        None,
    )
    assert show_at is not None, (
        f"{workflow_name}:{job_name} uses the compiler cache without reporting "
        "its statistics, so nobody can tell whether it worked"
    )
    builds = [
        index for index, step in enumerate(steps(definition)) if is_build_step(step)
    ]
    assert builds, f"{workflow_name}:{job_name} compiles nothing"
    assert max(builds) < show_at, (
        f"{workflow_name}:{job_name} reports its statistics before the build "
        "they are meant to describe"
    )
    report = scripts[show_at]
    assert "GITHUB_STEP_SUMMARY" in report and "tee" in report, (
        f"{workflow_name}:{job_name} must write its statistics to both the "
        "job summary and the log"
    )
    backend = f"steps.{SETUP_RUST_ID}.outputs.cache-backend"
    assert backend in str(steps(definition)[show_at].get("env", {})), (
        f"{workflow_name}:{job_name} must report {backend}; without it the "
        "statistics cannot be interpreted"
    )
