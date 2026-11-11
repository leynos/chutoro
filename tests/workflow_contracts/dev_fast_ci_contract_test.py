"""Contract tests for dev-fast CI workflow provisioning.

The Cranelift route exists for local edit-compile-test loops, so these
contracts pin that no merge gate selects or provisions it. The lint gate's
Clippy and Whitaker sequencing lives in `dev_fast_lint_serialization_test`."""

from __future__ import annotations

import re
import tomllib
from typing import Any

import pytest
import yaml

from dev_fast_workflow_scan_test import string_values
from workflow_support import ROOT

MAKEFILE_PATH = ROOT / "Makefile"
DEV_FAST_CONFIG = "tools/dev-fast/config.toml"
DEV_FAST_CONFIG_PATH = ROOT / DEV_FAST_CONFIG
WORKFLOW_PATHS = {
    "ci.yml": ROOT / ".github" / "workflows" / "ci.yml",
    "coverage-main.yml": ROOT / ".github" / "workflows" / "coverage-main.yml",
}


def _workflow(workflow_name: str) -> dict[str, Any]:
    """Return the loaded document for a workflow this contract owns.

    The whole document is returned, not only its jobs, because a workflow-level
    `env` mapping is merged into every step's environment just as a step-level
    one is.
    """
    workflow = yaml.safe_load(WORKFLOW_PATHS[workflow_name].read_text(encoding="utf-8"))
    assert isinstance(workflow, dict), f"{workflow_name} must be a mapping"
    return workflow


def _workflow_job(workflow_name: str, job_name: str) -> dict[str, Any]:
    """Return the named job mapping from a workflow this contract owns."""
    jobs = _workflow(workflow_name).get("jobs")
    assert isinstance(jobs, dict), f"{workflow_name} must define jobs"
    job = jobs.get(job_name)
    assert isinstance(job, dict), f"{workflow_name} must define job {job_name!r}"
    return job


def _step_named(job: dict[str, Any], name: str) -> tuple[int, dict[str, Any]]:
    """Return a job step by name, requiring the name to be unambiguous.

    The index comes back with the step so a caller can assert on ordering as
    well as content.
    """
    steps = job.get("steps")
    assert isinstance(steps, list), "workflow job must define a step list"
    matches = [
        (index, step) for index, step in enumerate(steps) if step.get("name") == name
    ]
    assert len(matches) == 1, f"expected one {name!r} step, found {len(matches)}"
    index, step = matches[0]
    assert isinstance(step, dict), f"step {name!r} must be a mapping"
    return index, step


#: Every way a workflow step can reach the development route. `dev-fast`
#: covers both the Makefile target prefix and the `tools/dev-fast` fragment
#: path, so a `--config` call cannot slip past a target-name-only check.
DEVELOPMENT_ROUTE_TOKENS = ("dev-build", "dev-test", "install-dev-fast", "dev-fast")


def _development_route_mentions(script: str) -> tuple[str, ...]:
    """Return the development-route tokens a step script contains.

    >>> _development_route_mentions("make dev-test")
    ('dev-test',)
    >>> _development_route_mentions("cargo --config tools/dev-fast/config.toml")
    ('dev-fast',)
    >>> _development_route_mentions("make lint")
    ()
    """
    return tuple(token for token in DEVELOPMENT_ROUTE_TOKENS if token in script)


def test_development_route_mentions_reports_every_matching_token() -> None:
    """A script that names a target and a path must report both.

    The scan drives the CI contract's provisioned-step check, so a token that
    went unreported would let that step through.
    """
    assert _development_route_mentions("cargo --config tools/dev-fast/config.toml") == (
        "dev-fast",
    )
    assert _development_route_mentions("make install-dev-fast && make dev-build") == (
        "dev-build",
        "install-dev-fast",
        "dev-fast",
    )


@pytest.mark.parametrize(
    ("workflow_name", "job_name"),
    (
        pytest.param("ci.yml", "build-test", id="pull-request-ci"),
        pytest.param("coverage-main.yml", "coverage-upload", id="main-coverage"),
    ),
)
def test_ci_never_runs_or_provisions_the_development_route(
    workflow_name: str, job_name: str
) -> None:
    """The Cranelift route is opt-in, so CI must never invoke it.

    The earlier rule inspected only the four-character-long target name
    `install-dev-fast`, which a step could satisfy while running
    `make dev-test`, or while selecting the fragment through a `--config` call.
    Every `run` script is scanned for all route tokens instead.
    """
    job = _workflow_job(workflow_name, job_name)
    _, lint_step = _step_named(job, "Lint")
    assert lint_step.get("run", "").strip() == "make lint", (
        f"{workflow_name} must run the standard lint gate, not a development one"
    )
    steps = job.get("steps")
    assert isinstance(steps, list), f"{workflow_name} job must define a step list"
    provisioned = [
        (step.get("name"), _development_route_mentions(script))
        for step in steps
        if isinstance(script := step.get("run"), str)
        if _development_route_mentions(script)
    ]
    assert not provisioned, (
        f"{workflow_name} must not provision the opt-in development route: "
        f"{provisioned!r}"
    )


def _dev_fast_config() -> dict[str, Any]:
    """Parse the pinned development fragment as Cargo itself will read it."""
    return tomllib.loads(DEV_FAST_CONFIG_PATH.read_text(encoding="utf-8"))


def test_development_fragment_enables_cranelift_for_the_dev_profile() -> None:
    """The fragment only shortens builds if Cargo is allowed to use Cranelift.

    `[unstable] codegen-backend` is the gate: without it Cargo parses the
    profile setting and silently compiles with LLVM instead, so the opt-in
    target would build slowly while looking correct.
    """
    config = _dev_fast_config()
    unstable = config.get("unstable")
    assert isinstance(unstable, dict), "the fragment must declare [unstable]"
    assert unstable.get("codegen-backend") is True, (
        "the fragment must enable Cargo's unstable codegen-backend support"
    )
    profile_dev = config.get("profile", {}).get("dev")
    assert isinstance(profile_dev, dict), "the fragment must declare [profile.dev]"
    assert profile_dev.get("codegen-backend") == "cranelift", (
        f"the development profile must select Cranelift: {profile_dev!r}"
    )


def test_development_fragment_keeps_the_linux_linker_policy() -> None:
    """The fragment must restate the standard flags, not silently drop them.

    Cargo selects a single `rustflags` source rather than merging them, so a
    Linux `[target.*]` table replaces `.cargo/config.toml`'s flags outright.
    A fragment that named the linker without repeating the documentation
    denials would make development builds compile that production builds
    reject.
    """
    linux = _dev_fast_config().get("target", {}).get('cfg(target_os = "linux")')
    assert isinstance(linux, dict), (
        "the fragment must declare a Linux target table for the mold linker"
    )
    assert linux.get("linker") == "gcc", (
        f"the fragment must select GCC so it finds ld.mold on PATH: {linux!r}"
    )
    rustflags = linux.get("rustflags")
    assert isinstance(rustflags, list), "the Linux table must set a rustflags list"
    for required in ("-Dmissing_docs", "-Dmissing_crate_level_docs"):
        assert required in rustflags, (
            f"the fragment's Linux rustflags lost {required!r}: {rustflags!r}"
        )
    assert "-Clink-arg=-fuse-ld=mold" in rustflags, (
        f"the fragment's Linux rustflags must select mold: {rustflags!r}"
    )


def test_development_toolchain_pin_names_a_nightly_release() -> None:
    """Cranelift needs nightly, and the pin must name one exact release.

    A floating `nightly` or `stable` pin would silently change compiler
    behaviour between runs; the fragment is only meaningful against a
    reproducible toolchain.
    """
    toolchain = (ROOT / "tools" / "dev-fast" / "TOOLCHAIN").read_text(encoding="utf-8")
    pin = toolchain.strip()
    assert re.fullmatch(r"nightly-\d{4}-\d{2}-\d{2}", pin), (
        f"the development toolchain pin must be one dated nightly: {pin!r}"
    )


def _dense_simd_commands(step: dict[str, Any]) -> list[str]:
    """Return the Cargo commands a step script runs, with continuations joined.

    Line continuations are folded first so a command split across source lines
    is compared whole rather than as fragments.
    """
    script = step.get("run")
    assert isinstance(script, str), "Dense stable SIMD gating must use a run script"
    joined = re.sub(r"\\\n[ \t]*", " ", script)
    return [
        line.strip()
        for line in joined.splitlines()
        if line.lstrip().startswith("cargo ")
    ]


@pytest.mark.parametrize(
    ("workflow_name", "job_name"),
    (
        pytest.param("ci.yml", "build-test", id="pull-request-ci"),
        pytest.param("coverage-main.yml", "coverage-upload", id="main-coverage"),
    ),
)
def test_ci_direct_simd_and_coverage_routes_stay_unfragmented(
    workflow_name: str, job_name: str
) -> None:
    """The direct SIMD and coverage routes must never select the fragment.

    Both jobs stay on the repository's stable pin, so a `--config` argument
    reaching either command would put a merge gate on the development route.
    """
    job = _workflow_job(workflow_name, job_name)
    _, dense_step = _step_named(job, "Dense stable SIMD gating")
    commands = _dense_simd_commands(dense_step)
    assert len(commands) == 2, (
        f"{workflow_name} must keep both direct dense SIMD Cargo commands: {commands!r}"
    )
    for command in commands:
        assert "--config" not in command and DEV_FAST_CONFIG not in command, (
            f"direct SIMD command must remain unfragmented: {command!r}"
        )

    _, coverage_step = _step_named(job, "Test and Measure Coverage")
    coverage_values = "\n".join(string_values(coverage_step))
    assert DEV_FAST_CONFIG not in coverage_values, (
        f"coverage action inputs must not select the dev-fast fragment: "
        f"{coverage_values!r}"
    )
    assert "--config" not in coverage_values, (
        f"coverage action inputs must not pass a Cargo config override: "
        f"{coverage_values!r}"
    )


def _cargo_home_selections(step: dict[str, Any]) -> tuple[str, ...]:
    """Return the values a step assigns to `CARGO_HOME`.

    A `--config` scan alone cannot see this route: Cargo reads
    `$CARGO_HOME/config.toml` before it parses a command line, so a step that
    points the home at a directory holding the fragment selects it with no
    `--config` argument anywhere. Both the `env:` mapping and a `NAME=value`
    prefix inside `run` are covered, because both reach Cargo the same way.

    >>> _cargo_home_selections({"env": {"CARGO_HOME": "/srv/cargo"}})
    ('/srv/cargo',)
    >>> _cargo_home_selections({"run": "CARGO_HOME=tools/dev-fast make test"})
    ('tools/dev-fast',)
    >>> _cargo_home_selections({"run": "make test"})
    ()
    """
    selections: list[str] = []
    environment = step.get("env")
    if isinstance(environment, dict):
        home = environment.get("CARGO_HOME")
        if isinstance(home, str):
            selections.append(home)
    run = step.get("run")
    if isinstance(run, str):
        selections.extend(
            value.strip("\"'") for value in re.findall(r"\bCARGO_HOME=(\S+)", run)
        )
    return tuple(selections)


@pytest.mark.parametrize(
    ("workflow_name", "job_name"),
    (
        pytest.param("ci.yml", "build-test", id="pull-request-ci"),
        pytest.param("coverage-main.yml", "coverage-upload", id="main-coverage"),
    ),
)
def test_ci_does_not_point_cargo_home_at_the_development_fragment(
    workflow_name: str, job_name: str
) -> None:
    """Nothing in the workflow may select the fragment through Cargo's home.

    A `--config` assertion says nothing about `CARGO_HOME`, which Cargo reads
    for `config.toml` before it ever parses a command line. Pointing the home
    at the fragment's directory would therefore enable Cranelift in a merge
    gate while every existing assertion still passed.

    All three scopes are read, because the environment a step sees is the
    merge of the workflow, the job, and the step. A scan of the step alone
    would pass while a job-level or workflow-level mapping selected the
    fragment, which is also the more usual place to put a shared variable.
    """
    workflow = _workflow(workflow_name)
    job = _workflow_job(workflow_name, job_name)
    steps = job.get("steps")
    assert isinstance(steps, list), f"{workflow_name} job must define a step list"

    selected = [
        (f"{workflow_name} env", selections)
        for selections in (_cargo_home_selections(workflow),)
        if selections
    ]
    selected += [
        (f"{workflow_name}/{job_name} env", selections)
        for selections in (_cargo_home_selections(job),)
        if selections
    ]
    selected += [
        (step.get("name"), selections)
        for step in steps
        if isinstance(step, dict)
        if (selections := _cargo_home_selections(step))
    ]

    assert not selected, (
        f"{workflow_name} must not set CARGO_HOME at any scope, because Cargo "
        f"would read a config.toml from it: {selected!r}"
    )
