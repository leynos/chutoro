"""Mutation contracts for the standard test gate's stable routing.

``make test`` is a merge gate, so these contracts pin it to the repository's
stable pin and fail if any development shortcut leaks into it.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from dev_fast_routing_test import MAKEFILE_PATH, _cargo_invocations, _make_output


REQUIRED_DENIALS = ("-D warnings", "-Dmissing_docs", "-Dmissing_crate_level_docs")


def _assert_stable_test(output: str) -> None:
    """Require the test gate to be one stable nextest run under the standard."""
    invocations = _cargo_invocations(output)
    assert len(invocations) == 1, (
        f"make test must emit exactly one Cargo invocation, got {invocations!r}"
    )
    line, tokens = invocations[0]
    # The RUSTFLAGS assignment prefixes the command, so read the whole line
    # rather than the token slice that begins at the cargo executable.
    assert "--config" not in shlex.split(line), (
        f"make test must not select a Cargo fragment: {line!r}"
    )
    assert tokens[1] == "nextest", (
        f"make test must run nextest on the stable pin: {line!r}"
    )
    assert not any("+nightly" in token for token in tokens), (
        f"make test must stay on the repository toolchain: {line!r}"
    )

    assignments = [
        token for token in shlex.split(line) if token.startswith("RUSTFLAGS=")
    ]
    assert len(assignments) == 1, (
        f"make test must assign RUSTFLAGS exactly once: {line!r}"
    )
    for required in REQUIRED_DENIALS:
        assert required in assignments[0], (
            f"make test RUSTFLAGS lost {required!r}: {line!r}"
        )
    assert "-Clink-arg=-fuse-ld=mold" in assignments[0], (
        f"make test must keep the Linux mold linker: {line!r}"
    )

    filter_index = tokens.index("-E")
    assert tokens[filter_index + 1] == "not kind(bench)", (
        f"make test must filter only benchmarks: {line!r}"
    )


def test_standard_test_is_a_single_stable_nextest_run() -> None:
    """The gate stays on the stable pin, under the build standard."""
    _assert_stable_test(_make_output("test", overrides=("BUILD_HOST_OS=Linux",)))


@pytest.mark.parametrize(
    ("old", "new", "failure"),
    (
        pytest.param(
            "$(CARGO) nextest run --profile $(NEXTEST_PROFILE)",
            "$(CARGO) --config tools/dev-fast/config.toml nextest run "
            "--profile $(NEXTEST_PROFILE)",
            "must not select a Cargo fragment",
            id="test-gains-dev-fast-fragment",
        ),
        pytest.param(
            "-D warnings $(STANDARD_RUSTFLAGS)",
            "-D warnings",
            "lost",
            id="test-loses-documentation-denials",
        ),
    ),
)
def test_test_contract_rejects_routing_mutations(
    old: str, new: str, failure: str, tmp_path: Path
) -> None:
    """Each required property of the stable gate is binding."""
    source = MAKEFILE_PATH.read_text(encoding="utf-8")
    assert source.count(old) == 1, f"mutation anchor must occur once: {old!r}"
    mutated = tmp_path / "Makefile"
    mutated.write_text(source.replace(old, new), encoding="utf-8")

    output = _make_output("test", makefile=mutated, overrides=("BUILD_HOST_OS=Linux",))
    with pytest.raises(AssertionError, match=failure):
        _assert_stable_test(output)


def test_make_test_propagates_a_failing_cargo(tmp_path: Path) -> None:
    """A failing nextest run must make the Make gate fail."""
    invocations_path = tmp_path / "cargo-invocations.txt"
    fake_cargo = tmp_path / "fake-cargo.py"
    fake_cargo.write_text(
        f"""import sys
from pathlib import Path

with Path({str(invocations_path)!r}).open("a", encoding="utf-8") as output:
    output.write(repr(sys.argv[1:]) + "\\n")
raise SystemExit(37)
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "--file",
            str(MAKEFILE_PATH),
            "test",
            f"CARGO={sys.executable} {fake_cargo}",
            "BUILD_HOST_OS=Darwin",
        ],
        cwd=MAKEFILE_PATH.parent,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0, (
        "make test must fail when its Cargo invocation fails"
    )
    invocations = invocations_path.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 1, (
        f"expected the single nextest invocation: {invocations!r}"
    )
    assert "nextest" in invocations[0], (
        f"the failing invocation was not nextest: {invocations!r}"
    )
