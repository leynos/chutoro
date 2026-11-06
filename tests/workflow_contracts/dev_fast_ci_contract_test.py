"""Contract tests for dev-fast CI workflow provisioning and lint sequencing."""

from __future__ import annotations

import os
import re
import subprocess
import textwrap
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from workflow_support import ROOT

MAKEFILE_PATH = ROOT / "Makefile"
DEV_FAST_CONFIG = "tools/dev-fast/config.toml"
DEV_FAST_CONFIG_PATH = ROOT / DEV_FAST_CONFIG
WORKFLOW_PATHS = {
    "ci.yml": ROOT / ".github" / "workflows" / "ci.yml",
    "coverage-main.yml": ROOT / ".github" / "workflows" / "coverage-main.yml",
}


def _workflow_job(workflow_name: str, job_name: str) -> dict[str, Any]:
    workflow = yaml.safe_load(WORKFLOW_PATHS[workflow_name].read_text(encoding="utf-8"))
    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict), f"{workflow_name} must define jobs"
    job = jobs.get(job_name)
    assert isinstance(job, dict), f"{workflow_name} must define job {job_name!r}"
    return job


def _step_named(job: dict[str, Any], name: str) -> tuple[int, dict[str, Any]]:
    steps = job.get("steps")
    assert isinstance(steps, list), "workflow job must define a step list"
    matches = [
        (index, step) for index, step in enumerate(steps) if step.get("name") == name
    ]
    assert len(matches) == 1, f"expected one {name!r} step, found {len(matches)}"
    index, step = matches[0]
    assert isinstance(step, dict), f"step {name!r} must be a mapping"
    return index, step


@pytest.mark.parametrize(
    ("workflow_name", "job_name"),
    (
        pytest.param("ci.yml", "build-test", id="pull-request-ci"),
        pytest.param("coverage-main.yml", "coverage-upload", id="main-coverage"),
    ),
)
def test_ci_never_provisions_the_development_fragment(
    workflow_name: str, job_name: str
) -> None:
    """The Cranelift route is opt-in, so CI must not install or select it."""
    job = _workflow_job(workflow_name, job_name)
    _, lint_step = _step_named(job, "Lint")
    assert lint_step.get("run", "").strip() == "make lint", (
        f"{workflow_name} must run the standard lint gate, not a development one"
    )
    steps = job.get("steps")
    assert isinstance(steps, list), f"{workflow_name} job must define a step list"
    provisioned = [
        step.get("run", "") for step in steps if "install-dev-fast" in step.get("run", "")
    ]
    assert not provisioned, (
        f"{workflow_name} must not provision the opt-in development fragment: "
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


def _string_values(value: Any) -> Iterator[str]:
    """Yield workflow scalar strings recursively for focused config checks.

    Containers are visited in their natural order and non-string scalars are
    skipped, so a reading of one step cannot silently reorder or invent values.

    >>> list(_string_values({"a": ["b", 1], "c": None}))
    ['b']
    >>> list(_string_values([1, ["x", {"y": "z"}]]))
    ['x', 'z']
    """
    match value:
        case str():
            yield value
        case dict():
            for nested in value.values():
                yield from _string_values(nested)
        case list():
            for nested in value:
                yield from _string_values(nested)


def test_string_values_preserves_container_order_and_omits_non_strings() -> None:
    workflow_value = {
        "first": "one",
        "nested": [
            "two",
            {"third": "three", "ignored": 7, "fourth": ["four", None]},
            False,
        ],
        "also_ignored": None,
    }

    assert list(_string_values(workflow_value)) == ["one", "two", "three", "four"], (
        "workflow value traversal must keep container order and drop non-strings"
    )


#: YAML-like scalars that are not strings. The traversal must ignore every one
#: of them, including the ones Python would happily print.
_NON_STRING_SCALARS = st.one_of(
    st.none(), st.booleans(), st.integers(), st.floats(allow_nan=False)
)


@st.composite
def _yaml_like_values(draw: st.DrawFn) -> Any:
    """Build one arbitrary YAML-shaped value for the traversal property.

    YAML mappings and sequences nest inside each other, which is exactly the
    recursion `_string_values` implements, so the generator descends instead of
    emitting a fixed depth.
    """
    return draw(
        st.recursive(
            st.text() | _NON_STRING_SCALARS,
            lambda children: st.lists(children) | st.dictionaries(st.text(), children),
            max_leaves=12,
        )
    )


def _collect_strings(value: Any) -> list[str]:
    """Collect the strings of a YAML-shaped value in depth-first order.

    Deliberately a second, differently shaped implementation. A property test
    that walked its input with the helper under test would agree with it by
    construction and prove nothing; this one states the order the contract
    promises in its own terms.

    >>> _collect_strings({"a": ["b", 1], "c": None})
    ['b']
    >>> _collect_strings([1, ["x", {"y": "z"}]])
    ['x', 'z']
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, (dict, list)):
        children = list(value.values()) if isinstance(value, dict) else list(value)
        return [text for child in children for text in _collect_strings(child)]
    return []


@given(value=_yaml_like_values())
def test_string_values_emits_exactly_the_nested_strings_in_depth_first_order(
    value: Any,
) -> None:
    """The traversal is an order-preserving, non-string-dropping projection.

    CodeRabbit asked for this as a property rather than more examples: the
    helper recurses over arbitrary YAML, so the invariant has to hold for
    nesting shapes nobody wrote a case for. Depth-first order is what makes
    the resulting scan readable as the workflow source.
    """
    assert list(_string_values(value)) == _collect_strings(value), (
        f"traversal of {value!r} must match an independent depth-first walk"
    )


def _dense_simd_commands(step: dict[str, Any]) -> list[str]:
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
    coverage_values = "\n".join(_string_values(coverage_step))
    assert (
        DEV_FAST_CONFIG not in coverage_values and "--config" not in coverage_values
    ), (
        f"coverage action inputs must not select the dev-fast fragment: "
        f"{coverage_values!r}"
    )


def _make_probe(path: Path, role: str) -> None:
    """Write a controlled fake gate that records overlap with its peer."""
    source = textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        import os
        from pathlib import Path
        import time

        state = Path(os.environ["DEV_FAST_PROBE_STATE"])
        role = {role!r}
        peer = "whitaker" if role == "cargo" else "cargo"
        lock = state / "lock"

        def acquire_lock():
            deadline = time.monotonic() + 5
            while True:
                try:
                    lock.mkdir()
                    return
                except FileExistsError:
                    if time.monotonic() >= deadline:
                        raise SystemExit("timed out acquiring probe lock")
                    time.sleep(0.005)

        acquire_lock()
        if (state / f"{{peer}}.running").exists():
            (state / "overlap").write_text(f"{{role}} overlapped {{peer}}\\n")
        (state / f"{{role}}.running").write_text("running\\n")
        lock.rmdir()

        time.sleep(0.15)

        acquire_lock()
        (state / f"{{role}}.running").unlink()
        (state / f"{{role}}.ran").write_text("ran\\n")
        lock.rmdir()
        """
    )
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def test_parallel_lint_serializes_clippy_and_whitaker(tmp_path: Path) -> None:
    cargo = tmp_path / "probe-cargo"
    whitaker = tmp_path / "probe-whitaker"
    state = tmp_path / "state"
    state.mkdir()
    _make_probe(cargo, "cargo")
    _make_probe(whitaker, "whitaker")

    environment = os.environ.copy()
    environment["DEV_FAST_PROBE_STATE"] = str(state)
    completed = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "-j2",
            "--file",
            str(MAKEFILE_PATH),
            "lint",
            f"CARGO={cargo}",
            f"WHITAKER={whitaker}",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, (
        f"controlled lint probe failed: {completed.stdout}\n{completed.stderr}"
    )
    # The absence of an overlap marker only means something if both probes
    # actually ran. A recipe that skipped `lint-clippy` or `lint-whitaker`
    # would trivially never overlap, so record each role's completion and
    # require both.
    for role in ("cargo", "whitaker"):
        assert (state / f"{role}.ran").exists(), (
            f"lint did not run the {role} probe under make -j2: "
            f"{sorted(path.name for path in state.iterdir())}"
        )
    assert not (state / "overlap").exists(), (
        "lint-clippy and lint-whitaker ran concurrently under make -j2"
    )
