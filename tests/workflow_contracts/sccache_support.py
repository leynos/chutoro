"""Shared vocabulary for the compiler-cache contracts.

The wiring contracts and the cache-entry contracts ask different questions
of the same workflows, and both need the same handful of names: which jobs
use the compiler cache, which backend each must demand, which forms of
hand-rolled wiring are retired. Restating them in each module is how two
contracts come to disagree about what the arrangement is.

The shared `setup-rust` action owns sccache here, as ADR 0005 in
`leynos/shared-actions` has it: it names the wrapper, starts and zeroes the
server, and selects the backend from the runner, Ubicloud's cache proxy on an
Ubicloud runner and a local directory it caches itself on a GitHub-hosted
one. The measurements behind the arrangement are in the developers guide,
under "The compiler cache".
"""

from __future__ import annotations

import functools
import re
import typing as typ

from workflow_support import (
    GITHUB_HOSTED_LABELS,
    conditional_runner,
    declared_cache_paths,
    jobs,
    load_workflow,
    run_script,
    runner_labels,
    steps,
    uses_reference,
    workflow_names,
)

#: The shared action that owns sccache, without its revision.
SETUP_RUST_PATH = "leynos/shared-actions/.github/actions/setup-rust"

#: The step id each cache job gives `setup-rust`, so its report can read the
#: selected backend from `steps.setup-rust.outputs.cache-backend`.
SETUP_RUST_ID = "setup-rust"

#: The jobs that use the compiler cache, mapped to the `expect-cache` value
#: each must pass. A list rather than a sweep, because a sweep over "jobs
#: that call setup-rust with sccache on" quietly shrinks to nothing when the
#: cache is switched off, which is the regression worth catching. The value
#: follows the placement: a job that can land on a GitHub-hosted runner, such
#: as a fork's pull request, must accept any backend, and a job that runs
#: only on Ubicloud demands the proxy so that its absence fails loudly.
CACHE_JOBS: dict[tuple[str, str], str] = {
    ("ci.yml", "build-test"): "any",
    ("coverage-main.yml", "coverage-upload"): "ubicloud",
}

#: The pull-request job whose compile shapes the trunk writer must cover,
#: and the push-to-main job that writes them into `main`'s cache scope.
EXPECTED_READER: tuple[str, str] = ("ci.yml", "build-test")
EXPECTED_WRITER: tuple[str, str] = ("coverage-main.yml", "coverage-upload")

#: Variables that configured the retired hand-rolled cache. `setup-rust`
#: sets or selects each of them, and a caller's value wins over its choice,
#: so any of these left in a workflow silently overrides the runner-aware
#: backend.
HAND_ROLLED_VARIABLES = (
    "RUSTC_WRAPPER",
    "SCCACHE_DIR",
    "SCCACHE_CACHE_SIZE",
    "SCCACHE_GHA_ENABLED",
)

#: The retired pinned-binary installer.
RETIRED_INSTALLER = "scripts/install-sccache.sh"

#: The shared action that republished Ubicloud's cache-proxy credentials.
#: `setup-rust` now exports them itself on an Ubicloud runner.
EXPORT_ACTION = (
    "leynos/shared-actions/.github/actions/export-ubicloud-cache-credentials@"
)

#: A `run:` command that starts or resets the server, which is `setup-rust`'s
#: job now. A second start from a script would bind whatever backend the
#: script's environment named.
SERVER_COMMAND = re.compile(r"\bsccache\s+--(?:zero-stats|start-server)\b")

#: A step that actually compiles something. Compilation also happens inside
#: shared actions, so those count as build steps too.
BUILD_COMMAND = re.compile(r"\bcargo\s+(nextest|test|build|clippy|llvm-cov)\b")
BUILD_ACTIONS = ("/actions/generate-coverage@", "/actions/rust-build-release@")


def is_build_step(step: dict[str, typ.Any]) -> bool:
    """Report whether a step compiles Rust."""
    if BUILD_COMMAND.search(run_script(step)):
        return True
    reference = uses_reference(step)
    return any(action in reference for action in BUILD_ACTIONS)


def expected_expect_cache(labels: typ.Iterable[str]) -> str:
    """Return the `expect-cache` value a job's placement calls for.

    Parameters
    ----------
    labels : Iterable[str]
        Every runner label the job can resolve to.

    Returns
    -------
    str
        ``"any"`` when a GitHub-hosted label is among them, else
        ``"ubicloud"``.

    >>> expected_expect_cache(["ubuntu-latest", "ubicloud-standard-2"])
    'any'
    >>> expected_expect_cache(["ubicloud-standard-2"])
    'ubicloud'
    """
    return "any" if set(labels) & GITHUB_HOSTED_LABELS else "ubicloud"


def _env_findings(scope: str, env: object) -> list[str]:
    """Return the hand-rolled variables one `env` mapping sets."""
    if not isinstance(env, dict):
        return []
    return [
        f"{scope} sets {name}" for name in HAND_ROLLED_VARIABLES if name in env
    ]


def _step_findings(index: int, step: dict[str, typ.Any]) -> list[str]:
    """Return the hand-rolled wiring one step carries."""
    findings = _env_findings(f"step {index}", step.get("env"))
    script = run_script(step)
    reference = uses_reference(step)
    if RETIRED_INSTALLER in script:
        findings.append(f"step {index} runs {RETIRED_INSTALLER}")
    if SERVER_COMMAND.search(script):
        findings.append(f"step {index} starts or zeroes the sccache server")
    if EXPORT_ACTION in reference:
        findings.append(f"step {index} exports the Ubicloud cache credentials")
    if reference.startswith("actions/cache") and any(
        "sccache" in path for path in declared_cache_paths(step)
    ):
        findings.append(f"step {index} archives a compiler-cache directory")
    return findings


def hand_rolled_findings(
    workflow_env: object, definition: dict[str, typ.Any]
) -> list[str]:
    """Return every piece of hand-rolled compiler-cache wiring in a job.

    Pure over the parsed data, so the rule can be exercised on fixtures as
    well as on the checked-in workflows.

    Parameters
    ----------
    workflow_env : object
        The workflow's top-level `env` mapping, or anything else when it
        declares none.
    definition : dict[str, Any]
        One parsed job.

    Returns
    -------
    list[str]
        One description per finding, empty for a job that leaves sccache to
        `setup-rust`.

    >>> hand_rolled_findings({}, {"steps": [{"uses": "actions/checkout@x"}]})
    []
    >>> hand_rolled_findings(
    ...     {"RUSTC_WRAPPER": "sccache"},
    ...     {"steps": [{"run": "sccache --zero-stats"}]},
    ... )
    ['workflow sets RUSTC_WRAPPER', 'step 0 starts or zeroes the sccache server']
    """
    findings = _env_findings("workflow", workflow_env)
    findings += _env_findings("job", definition.get("env"))
    for index, step in enumerate(steps(definition)):
        findings += _step_findings(index, step)
    return findings


def setup_rust_steps(definition: dict[str, typ.Any]) -> list[dict[str, typ.Any]]:
    """Return a job's `setup-rust` steps in declaration order."""
    return [
        step
        for step in steps(definition)
        if uses_reference(step).split("@", 1)[0] == SETUP_RUST_PATH
    ]


def job_labels(definition: dict[str, typ.Any]) -> list[str]:
    """Return every runner label a job can resolve to.

    A conditional `runs-on` contributes both arms, because which one a run
    lands on depends on the event that started it.
    """
    conditional = conditional_runner(definition)
    if conditional is not None:
        return [conditional.paid, conditional.otherwise]
    return runner_labels(definition)


#: One job, as (workflow file name, job name, job definition).
JobEntry = tuple[str, str, dict[str, typ.Any]]


@functools.cache
def all_jobs() -> tuple[JobEntry, ...]:
    """Return every job in every workflow, parsed once per session."""
    return tuple(
        (workflow_name, job_name, definition)
        for workflow_name in workflow_names()
        for job_name, definition in jobs(load_workflow(workflow_name)).items()
    )


def all_job_ids() -> list[str]:
    """Return stable identifiers for the whole-estate parametrization."""
    return [f"{workflow}:{name}" for workflow, name, _ in all_jobs()]


def step_index(
    definition: dict[str, typ.Any],
    predicate: typ.Callable[[dict[str, typ.Any]], bool],
) -> int | None:
    """Return the index of the first step satisfying a predicate."""
    return next(
        (index for index, step in enumerate(steps(definition)) if predicate(step)),
        None,
    )
