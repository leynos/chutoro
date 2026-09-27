"""Contract-test that `setup-rust` owns the compiler cache and nothing else does.

Installing sccache is not the same as using it, and using it is not the same
as storing anything. This repository has failed both ways, and both failures
reported success: first nothing named the wrapper, then sccache wrote past
Ubicloud's proxy to GitHub's cache service, which rejected 273 writes. The
workaround that followed, a pinned binary, a workspace directory and an
`actions/cache` archive, worked, but every job had to carry it by hand.

The shared `setup-rust` action now selects the backend from the runner (ADR
0005 in `leynos/shared-actions`): Ubicloud's proxy on an Ubicloud runner, a
local directory it restores and saves itself on a GitHub-hosted one. These
tests pin the switching-on and the hand-over. The cache jobs call it with
sccache on and demand the backend their placement allows, and no job keeps
any piece of the retired wiring, because a caller's `RUSTC_WRAPPER` or
`SCCACHE_DIR` silently overrides the action's choice. Which shapes the
trunk writes is `sccache_cache_entry_test`'s question.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import re
import typing as typ

import pytest
from sccache_support import (
    CACHE_JOBS,
    SETUP_RUST_ID,
    all_job_ids,
    all_jobs,
    expected_expect_cache,
    hand_rolled_findings,
    job_labels,
    setup_rust_steps,
)
from workflow_support import job, load_workflow, uses_reference

#: A full commit pin, the only form a shared action may take here.
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


@pytest.mark.parametrize(("workflow_name", "job_name"), list(CACHE_JOBS))
def test_the_cache_jobs_hand_sccache_to_setup_rust(
    workflow_name: str, job_name: str
) -> None:
    """One pinned `setup-rust` call, sccache on, and an id the report reads."""
    calls = setup_rust_steps(job(workflow_name, job_name))
    assert len(calls) == 1, (
        f"{workflow_name}:{job_name} must call setup-rust exactly once; "
        f"found {len(calls)}"
    )
    step = calls[0]
    _, _, revision = uses_reference(step).partition("@")
    assert FULL_SHA.match(revision), (
        f"{workflow_name}:{job_name} must pin setup-rust to a full commit "
        f"SHA, not {revision!r}"
    )
    with_block = step.get("with") or {}
    assert str(with_block.get("use-sccache", "true")) == "true", (
        f"{workflow_name}:{job_name} switches setup-rust's sccache off, so "
        "the job compiles with no cache at all"
    )
    assert step.get("id") == SETUP_RUST_ID, (
        f"{workflow_name}:{job_name} must give setup-rust the id "
        f"{SETUP_RUST_ID!r}, or its report cannot name the backend"
    )


@pytest.mark.parametrize(
    ("workflow_name", "job_name", "expected"),
    [(*key, value) for key, value in CACHE_JOBS.items()],
)
def test_each_cache_job_demands_the_backend_its_placement_allows(
    workflow_name: str, job_name: str, expected: str
) -> None:
    """`ubicloud` fails a proxy-less Ubicloud job loudly; `any` spares a fork.

    The reviewed value and the job's placement must agree. A job that can
    land on a GitHub-hosted runner, as a fork's pull request does, would fail
    every such run under `ubicloud`; a job that cannot would compile against
    local disk unnoticed under `any` whenever the proxy went missing.
    """
    definition = job(workflow_name, job_name)
    assert expected == expected_expect_cache(job_labels(definition)), (
        f"{workflow_name}:{job_name} is reviewed as expect-cache {expected!r}, "
        f"but its runners {job_labels(definition)} call for "
        f"{expected_expect_cache(job_labels(definition))!r}"
    )
    with_block = setup_rust_steps(definition)[0].get("with") or {}
    assert with_block.get("expect-cache") == expected, (
        f"{workflow_name}:{job_name} must pass expect-cache: {expected}, "
        f"not {with_block.get('expect-cache')!r}"
    )


@pytest.mark.parametrize(
    ("workflow_name", "job_name", "definition"),
    all_jobs(),
    ids=all_job_ids(),
)
def test_no_job_hand_rolls_the_compiler_cache(
    workflow_name: str, job_name: str, definition: dict[str, typ.Any]
) -> None:
    """Every retired piece stays retired, in every job, not just the cache jobs.

    A caller's wrapper or directory wins over `setup-rust`'s choice, a
    script that starts the server binds whatever its environment names, the
    credentials export is `setup-rust`'s own work now, and an archived
    compiler-cache directory would be a second owner beside the backend the
    action selected.
    """
    workflow_env = load_workflow(workflow_name).get("env")
    findings = hand_rolled_findings(workflow_env, definition)
    assert not findings, (
        f"{workflow_name}:{job_name} still hand-rolls the compiler cache: "
        + "; ".join(findings)
    )


@pytest.mark.parametrize(
    ("definition", "expected"),
    [
        pytest.param({"env": {"SCCACHE_DIR": "x"}, "steps": []}, 1, id="job-dir"),
        pytest.param(
            {"steps": [{"run": "scripts/install-sccache.sh"}]}, 1, id="installer"
        ),
        pytest.param(
            {"steps": [{"run": "sccache --start-server"}]}, 1, id="server-start"
        ),
        pytest.param(
            {
                "steps": [
                    {
                        "uses": "leynos/shared-actions/.github/actions/"
                        "export-ubicloud-cache-credentials@abc"
                    }
                ]
            },
            1,
            id="credentials-export",
        ),
        pytest.param(
            {
                "steps": [
                    {
                        "uses": "actions/cache/restore@abc",
                        "with": {"path": "${{ github.workspace }}/.sccache"},
                    }
                ]
            },
            1,
            id="archived-directory",
        ),
        pytest.param(
            {"steps": [{"env": {"SCCACHE_GHA_ENABLED": "true"}, "run": "make"}]},
            1,
            id="step-backend-switch",
        ),
        pytest.param(
            {"steps": [{"run": '"$SCCACHE_PATH" --show-stats'}]}, 0, id="report"
        ),
        pytest.param(
            {
                "steps": [
                    {
                        "uses": "actions/cache@abc",
                        "with": {"path": "~/.cargo/registry"},
                    }
                ]
            },
            0,
            id="registry-cache",
        ),
    ],
)
def test_the_hand_rolled_reader_is_narrow_as_well_as_sufficient(
    definition: dict[str, typ.Any], expected: int
) -> None:
    """The reader flags each retired form and leaves the permitted ones alone.

    Reading the checked-in workflows can only show that the rule passes on
    them. These fixtures show that it would catch each retired form, and that
    it does not catch the statistics report or an unrelated cache, which a
    rule matching the bare word `sccache` would.
    """
    assert len(hand_rolled_findings({}, definition)) == expected
