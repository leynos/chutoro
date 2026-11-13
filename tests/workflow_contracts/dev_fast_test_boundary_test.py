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

from dev_fast_make_support import MAKEFILE_PATH, _cargo_invocations, _make_output


REQUIRED_DENIALS = ("-D warnings", "-Dmissing_docs", "-Dmissing_crate_level_docs")

#: The text the mutation removes to model a gate that dropped the build
#: standard. It reaches into the nextest recipe rather than stopping at the
#: flags, because the same flag string legitimately appears on other recipes:
#: `test-doc` repeats it, and a mutation anchored on the flags alone stopped
#: finding exactly one occurrence once trunk gained that target. Naming the
#: command keeps the anchor unique without weakening what the mutation proves.
STANDARD_RUSTFLAGS_ANCHOR = (
    '-D warnings $(STANDARD_RUSTFLAGS)" $(CARGO) nextest run'
)


def _assert_stable_test(output: str) -> None:
    """Require the test gate to be one stable nextest run under the standard.

    Trunk may prefix the run with preparatory Cargo commands - `make test`
    begins with `$(CARGO) fetch --locked &&` - so the contract requires exactly
    one *nextest* invocation rather than exactly one command, while still
    rejecting a fragment selection or a toolchain override anywhere in the
    recipe.

    The gate must also keep *every* test case it can run. The poisoned-lock
    regression panics inside `catch_unwind`; that is fine on the stable pin
    and fatal on the Cranelift development route, so the exclusion belongs to
    `dev-test` alone. A filter that grew one here would silently retire the
    regression at the merge gate.
    """
    invocations = _cargo_invocations(output)
    assert invocations, f"make test must emit Cargo commands, got {output!r}"
    for line, tokens in invocations:
        assert "--config" not in shlex.split(line), (
            f"make test must not select a Cargo fragment: {line!r}"
        )
        assert not any("+nightly" in token for token in tokens), (
            f"make test must stay on the repository toolchain: {line!r}"
        )

    nextest = [(line, tokens) for line, tokens in invocations if tokens[1] == "nextest"]
    assert len(nextest) == 1, (
        f"make test must run nextest exactly once, got {nextest!r}"
    )
    line, tokens = nextest[0]

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
    expression = tokens[filter_index + 1]
    assert "test(=" not in expression, (
        f"make test must not exclude a named case: {line!r}"
    )
    assert expression == "not kind(bench)", (
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
            STANDARD_RUSTFLAGS_ANCHOR,
            '-D warnings" $(CARGO) nextest run',
            "lost",
            id="test-loses-documentation-denials",
        ),
        pytest.param(
            "-E 'not kind(bench)'",
            "-E '$(DEV_TEST_FILTER)'",
            "must not exclude a named case",
            id="test-inherits-the-development-exclusion",
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
    # The recipe may chain commands, and the first one is the one that fails,
    # so the evidence needed is that Cargo ran and the gate failed - not that
    # the failing command happened to be nextest.
    invocations = invocations_path.read_text(encoding="utf-8").splitlines()
    assert invocations, "make test did not reach the injected Cargo executable"
    assert "fetch" in invocations[0] or "nextest" in invocations[0], (
        f"the gate failed without a recognised Cargo command: {invocations!r}"
    )
