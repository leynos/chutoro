"""The property lanes select the profile sized for their workload.

Both lanes share one anchored run step, so the selected nextest profile is
the only difference between them. The pull-request lane runs 250 cases and
keeps the CI allowance; the weekly and manual lane runs 25,000 forked cases
and selects ``property-heavy``, which inherits ``ci`` and raises only the
dense SIMD parity allowance. Pinning the selection here means an edit that
moves the shared step back to one profile fails rather than silently running
the weekly workload under the pull-request budget (#264).

See "Dense SIMD parity suite" in ``docs/developers-guide.md``.
"""

import typing as typ

import pytest
from nextest_durations import _seconds
from timeout_budgets import (
    NEXTEST_CONFIG,
    _optional_seconds,
    parse_config,
    resolved_global_timeout,
    termination_allowance_of,
)
from workflow_support import job, run_script, steps

WORKFLOW: typ.Final[str] = "property-tests.yml"
PROFILE_VARIABLE: typ.Final[str] = "NEXTEST_PROFILE"
DENSE_PARITY_FILTER_FRAGMENT: typ.Final[str] = "simd::tests::parity::"
PR_JOB: typ.Final[str] = "property-tests-pr"
WEEKLY_JOB: typ.Final[str] = "property-tests-weekly"
PR_PROFILE: typ.Final[str] = "ci"
WEEKLY_PROFILE: typ.Final[str] = "property-heavy"

#: Time the weekly lane spends outside nextest's own clock: checkout, the
#: Rust toolchain, the nextest download, and the cold build of the workspace
#: and its dependencies before the first test starts. Thirty minutes is well
#: above the setup and compile phases measured on the forced measurement runs,
#: and below the inherited whole-run budget it is added to, so it bounds the
#: ceiling's headroom rather than consuming it.
PROPERTY_LANE_SETUP_ALLOWANCE_SECONDS: typ.Final[float] = 30 * 60.0


def _job_environment(job_name: str) -> dict[str, typ.Any]:
    """Return a property job's job-level environment mapping.

    Returns
    -------
    dict[str, typing.Any]
        The parsed ``env`` mapping.

    Raises
    ------
    AssertionError
        If the job declares no job-level environment. The shared run step
        reads the profile from there, so without one it expands to an empty
        ``--profile`` argument and nextest refuses the invocation.
    """
    environment = job(WORKFLOW, job_name).get("env")
    assert isinstance(environment, dict), (
        f"{job_name} must declare a job-level env carrying {PROFILE_VARIABLE}"
    )
    return environment


def _nextest_run_scripts(job_name: str) -> list[str]:
    """Return every run script in a job that invokes ``cargo nextest``.

    Returns
    -------
    list[str]
        The matching scripts.
    """
    return [
        script
        for script in (run_script(step) for step in steps(job(WORKFLOW, job_name)))
        if "cargo nextest run" in script
    ]


def _dense_parity_allowance(config_text: str, profile: str) -> float:
    """Return a profile's dense parity per-test allowance, in seconds.

    Returns
    -------
    float
        The allowance, being the period times the number of periods nextest
        waits before terminating.

    Raises
    ------
    AssertionError
        If the profile declares no single matching override or no
        terminating ``slow-timeout``. A test with no terminating override
        runs unbounded, which is the shape this contract exists to refuse.
    """
    section = parse_config(config_text)["profile"][profile]
    overrides = section.get("overrides")
    assert isinstance(overrides, list), (
        f"{profile} must declare overrides for the dense parity suite"
    )
    matching = [
        override
        for override in overrides
        if isinstance(override, dict)
        and DENSE_PARITY_FILTER_FRAGMENT in str(override.get("filter", ""))
    ]
    assert len(matching) == 1, (
        f"{profile} must declare exactly one dense parity override; found "
        f"{len(matching)}"
    )
    timeout = matching[0].get("slow-timeout")
    assert isinstance(timeout, dict), f"{profile} must declare a slow-timeout table"
    terminate_after = timeout.get("terminate-after")
    assert isinstance(terminate_after, int) and not isinstance(terminate_after, bool), (
        f"{profile}'s dense parity slow-timeout must terminate the test"
    )
    return _seconds(str(timeout["period"])) * float(terminate_after)


@pytest.mark.parametrize(
    ("job_name", "profile"),
    [
        pytest.param(PR_JOB, PR_PROFILE, id="pull-request-lane"),
        pytest.param(WEEKLY_JOB, WEEKLY_PROFILE, id="weekly-and-manual-lane"),
    ],
)
def test_property_jobs_select_the_profile_sized_for_their_workload(
    job_name: str, profile: str
) -> None:
    """The shared step must read the profile from the job that runs it.

    A job-level variable is what lets one anchored step serve both lanes.
    Checking only that the variable exists would accept a job that sets it
    and a step that ignores it, so every nextest invocation is required to
    name the variable and no invocation may hard-code a profile.
    """
    environment = _job_environment(job_name)
    assert environment.get(PROFILE_VARIABLE) == profile, (
        f"{job_name} must select {PROFILE_VARIABLE}={profile!r}"
    )
    scripts = _nextest_run_scripts(job_name)
    assert scripts, f"{job_name} must invoke cargo nextest"
    for script in scripts:
        assert f'--profile "${PROFILE_VARIABLE}"' in script, (
            f"{job_name}'s nextest invocation must read {PROFILE_VARIABLE}: "
            f"{script}"
        )


def test_the_heavy_dense_allowance_stays_inside_the_global_timeout() -> None:
    """A per-test allowance at or above the whole-run budget cannot be used.

    ``property-heavy`` declares no ``global-timeout`` of its own, so the
    resolved value is the one it inherits through ``ci``. Reading only the
    selected profile's own table would report no budget at all and turn this
    comparison into a skipped one.
    """
    config_text = NEXTEST_CONFIG.read_text(encoding="utf-8")
    allowance = _dense_parity_allowance(config_text, WEEKLY_PROFILE)
    budget = resolved_global_timeout(config_text, WEEKLY_PROFILE)
    assert allowance < budget, (
        f"the {allowance:.0f}s heavy dense allowance is not below the "
        f"{budget:.0f}s global-timeout it inherits; the run would end before "
        f"the test could use its budget"
    )


def test_the_weekly_job_ceiling_covers_the_heavy_run_and_its_setup() -> None:
    """The job timer is the outermost tier and must not pre-empt the inner one.

    nextest starts its whole-run clock only once tests begin, while the job
    timer starts with checkout and covers the cold build as well. A ceiling
    merely above the whole-run budget is still pre-empting it whenever setup
    and compilation take longer than the difference.
    """
    config_text = NEXTEST_CONFIG.read_text(encoding="utf-8")
    budget = resolved_global_timeout(config_text, WEEKLY_PROFILE)
    termination = termination_allowance_of(config_text)
    ceiling = _optional_seconds(job(WORKFLOW, WEEKLY_JOB).get("timeout-minutes"))
    assert ceiling is not None, f"{WEEKLY_JOB} must declare timeout-minutes"
    required = budget + termination + PROPERTY_LANE_SETUP_ALLOWANCE_SECONDS
    assert ceiling > required, (
        f"{WEEKLY_JOB} has a {ceiling:.0f}s ceiling, not above the "
        f"{required:.0f}s needed for the {budget:.0f}s inherited whole-run "
        f"budget, {termination:.0f}s for nextest to terminate the run, and "
        f"{PROPERTY_LANE_SETUP_ALLOWANCE_SECONDS:.0f}s of setup and cold build"
    )