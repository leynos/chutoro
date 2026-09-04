"""Contract-test that the compiler cache is actually wired up.

Installing sccache is not the same as using it. `setup-rust` runs
`mozilla-actions/sccache-action`, which provides the binary, but nothing
names it as the compiler wrapper, so for a long time every compilation
bypassed it while the job logs happily reported a working cache with zero
compile requests. That went unnoticed until the shared action stopped
archiving a `target` tree and the coverage gate grew by four minutes.

These tests pin the three parts that make the cache real: the wrapper is
named, the counters are zeroed before the build so the statistics describe
this run, and the statistics reach the job summary where an operator can
read them. They also pin the Ubicloud credential export, which must run
before `setup-rust` starts the sccache server and must never run on a
GitHub-hosted job.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import re
import typing as typ

import pytest
from workflow_support import (
    GITHUB_HOSTED_LABELS,
    job,
    jobs,
    load_workflow,
    run_script,
    runner_labels,
    steps,
    uses_reference,
    workflow_names,
)

#: The wrapper every instrumented job must name. sccache-action puts the
#: binary on PATH, so the bare name resolves.
EXPECTED_WRAPPER = "sccache"

#: The jobs that must name the wrapper. This is a list rather than a sweep
#: because a sweep over "jobs that set RUSTC_WRAPPER" quietly shrinks to
#: nothing when someone removes the variable, which is the exact regression
#: worth catching. The compile-heavy gates and the property suites are in
#: scope; `verus-proofs` and `nightly-portable-simd` do not use Setup Rust and
#: so have no sccache to name, and `nightly-kani` compiles through
#: `kani-compiler`, which sccache does not support.
WRAPPER_REQUIRED = (
    ("ci.yml", "build-test"),
    ("coverage-main.yml", "coverage-upload"),
    ("property-tests.yml", "property-tests-pr"),
    ("property-tests.yml", "property-tests-weekly"),
)

#: A step that actually compiles something. The statistics are meaningless
#: unless one of these runs between the reset and the report. Compilation
#: also happens inside shared actions, so those count as build steps too.
BUILD_COMMAND = re.compile(r"\bcargo\s+(nextest|test|build|clippy|llvm-cov)\b")
BUILD_ACTIONS = ("/actions/generate-coverage@", "/actions/rust-build-release@")


def _is_build_step(step: dict[str, typ.Any]) -> bool:
    """Report whether a step compiles Rust."""
    if BUILD_COMMAND.search(run_script(step)):
        return True
    reference = uses_reference(step)
    return any(action in reference for action in BUILD_ACTIONS)


#: The shared action that republishes Ubicloud's cache-proxy credentials.
EXPORT_ACTION = (
    "leynos/shared-actions/.github/actions/export-ubicloud-cache-credentials@"
)


def _wrapper_jobs() -> list[tuple[str, str, dict[str, typ.Any]]]:
    """Return every job that names a compiler wrapper."""
    return [
        (workflow_name, job_name, definition)
        for workflow_name in workflow_names()
        for job_name, definition in jobs(load_workflow(workflow_name)).items()
        if isinstance(definition.get("env"), dict)
        and "RUSTC_WRAPPER" in definition["env"]
    ]


def _wrapper_job_ids() -> list[str]:
    """Return stable identifiers for the wrapper-job parametrization."""
    return [f"{workflow}:{name}" for workflow, name, _ in _wrapper_jobs()]


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
    assert build_at is not None and build_at < show_at, (
        f"{workflow_name}:{job_name} must compile between zeroing its "
        "counters and reporting them, or the statistics describe no work"
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


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_the_cache_proxy_export_is_ubicloud_only(workflow_name: str) -> None:
    """The export fails closed off Ubicloud, so placing it there breaks the job."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        exports = [
            index
            for index, step in enumerate(steps(definition))
            if EXPORT_ACTION in uses_reference(step)
        ]
        if not exports:
            continue
        labels = runner_labels(definition)
        assert all(label not in GITHUB_HOSTED_LABELS for label in labels), (
            f"{workflow_name}:{job_name} runs on {labels} and must not export "
            "Ubicloud cache credentials"
        )


def test_the_paid_job_exports_the_proxy_credentials_before_setup() -> None:
    """`setup-rust` starts the sccache server, so the export has to precede it.

    The runner hands its cache-proxy URL and token to action steps only.
    Republishing them after the server has started leaves it pointed at the
    wrong endpoint for the rest of the job.
    """
    job_steps = steps(job("property-tests.yml", "property-tests-pr"))
    references = [uses_reference(step) for step in job_steps]
    export_at = next(
        (index for index, ref in enumerate(references) if EXPORT_ACTION in ref), None
    )
    checkout_at = next(
        (
            index
            for index, ref in enumerate(references)
            if ref.startswith("actions/checkout@")
        ),
        None,
    )
    setup_at = next(
        (
            index
            for index, ref in enumerate(references)
            if "/actions/setup-rust@" in ref
        ),
        None,
    )
    assert export_at is not None, (
        "property-tests-pr runs on Ubicloud and must export its cache-proxy "
        "credentials, or sccache cannot reach the proxy"
    )
    assert checkout_at is not None and setup_at is not None
    assert checkout_at < export_at < setup_at, (
        "the export must sit between checkout and Setup Rust; found order "
        f"checkout={checkout_at}, export={export_at}, setup-rust={setup_at}"
    )
