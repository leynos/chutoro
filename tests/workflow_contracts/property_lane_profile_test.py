"""The property lanes select the profile sized for their workload.

All lanes share one anchored run step, so the selected nextest profile is the
only difference between them. The pull-request lane runs 250 cases and keeps
the CI allowance. The weekly and manual lane runs 25,000 forked cases, and
only its ``dense_simd`` leg selects ``property-heavy``, which raises the dense
SIMD parity allowance on measurement. The other weekly legs stay on ``ci``:
the pinned cargo-nextest 0.9.143 consults only the selected profile's own
overrides and ``profile.default``'s, so a profile inheriting ``ci`` silently
drops ``ci``'s 1500-second HNSW allowances (fixed in 0.9.145).

Pinning the selection here means an edit that moves the shared step back to
one profile, or that sends another leg through the heavy profile, fails
rather than silently changing a lane's budget (#264).

See "Dense SIMD parity suite" in ``docs/developers-guide.md``.
"""

import typing as typ

from nextest_durations import _seconds
from profile_inheritance import resolved_global_timeout
from timeout_budgets import (
    NEXTEST_CONFIG,
    _optional_seconds,
    parse_config,
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
MATRIX_PROFILE_EXPRESSION: typ.Final[str] = "${{ matrix.nextest_profile }}"

#: Time the weekly lane spends outside nextest's own clock: checkout, the
#: Rust toolchain, the nextest download, and the cold build of the workspace
#: and its dependencies before the first test starts. Measured at 61 s from
#: job start to nextest start on run 44915040254; thirty minutes bounds a cold
#: runner image, toolchain download and full rebuild, and still leaves the
#: inherited whole-run budget plus the rest of the job below the 7200 s
#: ceiling.
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


#: The text every nextest invocation must begin with. Checking the whole
#: script instead would pass whenever any one of its four commands named the
#: variable, leaving a branch that hard-codes a profile invisible.
REQUIRED_PROFILE_ARGUMENT: typ.Final[str] = f'--profile "${PROFILE_VARIABLE}"'


def _nextest_invocations(job_name: str) -> list[str]:
    """Return the text each nextest invocation begins with.

    Returns
    -------
    list[str]
        One entry per invocation, with line continuations joined so a flag
        split across lines is still read.
    """
    return [
        fragment.strip()
        for script in _nextest_run_scripts(job_name)
        for fragment in script.replace("\\\n", " ").split("cargo nextest run")[1:]
    ]


def _hard_coded_invocations(scripts: list[str]) -> list[str]:
    """Return the nextest invocations that do not read the variable.

    Parameters
    ----------
    scripts : list[str]
        Run scripts to inspect.

    Returns
    -------
    list[str]
        One entry per offending invocation.
    """
    return [
        fragment.strip()
        for script in scripts
        for fragment in script.replace("\\\n", " ").split("cargo nextest run")[1:]
        if not fragment.strip().startswith(REQUIRED_PROFILE_ARGUMENT)
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


def _matrix_profiles(job_name: str) -> dict[str, str]:
    """Return the nextest profile each matrix leg selects.

    Returns
    -------
    dict[str, str]
        Suite name to profile name.

    Raises
    ------
    AssertionError
        If the job's matrix is not an ``include`` list of suites carrying a
        profile. A missing profile would expand the shared step's
        ``--profile`` argument to nothing.
    """
    matrix = job(WORKFLOW, job_name).get("strategy", {}).get("matrix")
    assert isinstance(matrix, dict), f"{job_name} must declare a matrix"
    include = matrix.get("include")
    assert isinstance(include, list), (
        f"{job_name}'s matrix must be an include list carrying nextest_profile"
    )
    profiles: dict[str, str] = {}
    for entry in include:
        assert isinstance(entry, dict), f"{job_name}'s matrix entries must be mappings"
        suite = entry.get("suite")
        profile = entry.get("nextest_profile")
        assert isinstance(suite, str) and isinstance(profile, str), (
            f"{job_name}'s matrix entry must name a suite and a profile: {entry}"
        )
        profiles[suite] = profile
    return profiles


def _assert_shared_step_reads_the_variable(job_name: str) -> None:
    """Require every nextest invocation in a job to name the variable.

    Checking only that the variable exists would accept a job that sets it
    and a step that hard-codes a profile. Checking each script as a whole
    would accept a step whose four commands carry the variable on three and
    a literal on the fourth, so each invocation is checked separately.
    """
    scripts = _nextest_run_scripts(job_name)
    assert scripts, f"{job_name} must invoke cargo nextest"
    invocations = _nextest_invocations(job_name)
    assert len(invocations) == 4, (
        f"{job_name}'s shared step must declare the four suite branches; "
        f"found {len(invocations)} nextest invocations"
    )
    offenders = _hard_coded_invocations(scripts)
    assert not offenders, (
        f"{job_name}'s nextest invocations must read {PROFILE_VARIABLE}: "
        f"{offenders}"
    )


def test_a_single_hard_coded_profile_is_reported() -> None:
    """One variable-based command must not hide one hard-coded command.

    The whole-script check this replaces passed whenever any command named
    the variable, so a branch that hard-coded a profile was invisible.
    """
    script = (
        'cargo nextest run --profile "$NEXTEST_PROFILE" -p a \\\n'
        "  a_test\n"
        "cargo nextest run --profile ci -p a \\\n"
        "  another_test\n"
    )
    offenders = _hard_coded_invocations([script])
    assert len(offenders) == 1, offenders
    assert offenders[0].startswith("--profile ci"), offenders


def test_the_pull_request_lane_selects_the_ci_profile() -> None:
    """The 250-case lane keeps the profile every other CI run uses."""
    environment = _job_environment(PR_JOB)
    assert environment.get(PROFILE_VARIABLE) == PR_PROFILE, (
        f"{PR_JOB} must select {PROFILE_VARIABLE}={PR_PROFILE!r}"
    )
    _assert_shared_step_reads_the_variable(PR_JOB)


def test_only_the_dense_weekly_leg_selects_the_heavy_profile() -> None:
    """A profile inheriting ``ci`` drops its overrides on nextest 0.9.143.

    The heavy profile is the only one whose own override matters to this
    lane. Sending the HNSW or edge-harvest legs through it would silently
    replace their 1500-second and 600-second allowances with the default
    profile's, because the pinned nextest does not consult an intermediate
    profile's overrides.
    """
    environment = _job_environment(WEEKLY_JOB)
    assert environment.get(PROFILE_VARIABLE) == MATRIX_PROFILE_EXPRESSION, (
        f"{WEEKLY_JOB} must take {PROFILE_VARIABLE} from its matrix"
    )
    assert _matrix_profiles(WEEKLY_JOB) == {
        "hnsw": "ci",
        "edge_harvest": "ci",
        "mst": "ci",
        "dense_simd": WEEKLY_PROFILE,
    }, "only the dense SIMD leg may select the heavy profile"
    _assert_shared_step_reads_the_variable(WEEKLY_JOB)


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