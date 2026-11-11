"""Executed contracts for the development test route's Nextest filter.

`dev_fast_routing_test` evaluates the Makefile dry run and compares the emitted
filter expression as text. That proves the recipe carries the intended
expression, but not that Nextest selects the intended cases from the real test
set. These contracts run Nextest's own filter evaluation over the workspace and
compare it with an unfiltered baseline, so a filter that Nextest accepts but
applies differently from the intended exclusion is caught.

Listing compiles the test binaries but runs no test. The filter is evaluated
over the compiled test names and target kinds, which the codegen backend does
not change, so the check runs on the repository's stable pin. The Cranelift
route stays opt-in and out of the merge gate, as `docs/developers-guide.md`
requires.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from dev_fast_make_support import REPO_ROOT, _make_output, _nextest_filter_expression
from dev_fast_routing_test import POISONED_LOCK_TEST


def _nextest_listing(filter_expression: str | None) -> dict[str, Any]:
    """Return the parsed `cargo nextest list` report for one filter.

    The listing is the same workspace, target and feature selection the
    development route runs, so the filter is evaluated against the real test
    set rather than a sample.
    """
    command = [
        "cargo",
        "nextest",
        "list",
        "--workspace",
        "--all-targets",
        "--all-features",
        "--message-format",
        "json",
    ]
    if filter_expression is not None:
        command.extend(("-E", filter_expression))
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"cargo nextest list failed with {completed.returncode} "
        "(cargo-nextest must be installed to run this contract): "
        f"{completed.stderr[-2000:]}"
    )
    return json.loads(completed.stdout)


def _cases(report: dict[str, Any]) -> dict[tuple[str, str], tuple[str, bool, str]]:
    """Map each listed case to its target kind, ignored flag and filter status.

    Nextest reports a filtered listing as the full set with a `filter-match`
    status per case, so the status is what distinguishes a selected case from
    a rejected one.
    """
    return {
        (suite_id, name): (
            suite.get("kind", ""),
            testcase.get("ignored", False),
            testcase.get("filter-match", {}).get("status", "matches"),
        )
        for suite_id, suite in report["rust-suites"].items()
        for name, testcase in suite["testcases"].items()
    }


def test_development_filter_selects_exactly_the_intended_cases() -> None:
    """Nextest must select the suite minus benchmarks and the poisoned-lock case.

    The baseline is unfiltered so the exclusions cannot pass because the cases
    were absent: a renamed regression or a removed benchmark fails the
    precondition assertions instead of silently satisfying the exclusion.
    """
    output = _make_output("dev-test", overrides=("DEV_FAST_HOST_OS=Linux",))
    expression = _nextest_filter_expression(output)

    baseline = _cases(_nextest_listing(None))
    selected = _cases(_nextest_listing(expression))

    assert baseline, "the unfiltered listing must not be empty"
    assert set(selected) <= set(baseline), (
        "the filter must not invent cases the unfiltered listing does not have"
    )

    poisoned = [key for key in baseline if key[1] == POISONED_LOCK_TEST]
    assert len(poisoned) == 1, (
        f"the baseline must contain {POISONED_LOCK_TEST} exactly once: {poisoned!r}"
    )
    poisoned_key = poisoned[0]
    poisoned_status = selected.get(poisoned_key, ("", False, "absent"))[2]
    assert poisoned_status == "mismatch", (
        "the development filter must reject the poisoned-lock regression, "
        f"got {poisoned_status!r}"
    )

    benchmarks = {key for key, (kind, _, _) in baseline.items() if kind == "bench"}
    assert benchmarks, "the baseline must contain benchmark cases"
    assert not benchmarks & set(selected), (
        "the development filter must not select benchmark cases"
    )

    matched = {key for key, (_, _, status) in selected.items() if status == "matches"}
    runnable = {
        key
        for key, (kind, ignored, _) in baseline.items()
        if kind != "bench" and not ignored
    }
    assert matched == runnable - {poisoned_key}, (
        "the development route must select every other runnable case"
    )
