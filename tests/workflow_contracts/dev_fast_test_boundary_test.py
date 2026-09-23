"""Mutation contracts for stable byte-exact Make test routing."""

from pathlib import Path
import subprocess
import sys

import pytest

from dev_fast_routing_test import (
    MAKEFILE_PATH,
    _assert_test_boundary,
    _make_output,
    _mutated_makefile,
)


@pytest.mark.parametrize(
    ("old", "new", "failure"),
    (
        pytest.param(
            '\tRUSTFLAGS="$(STABLE_VERIFY_RUSTFLAGS)" env -u RUSTUP_TOOLCHAIN '
            '$(CARGO) test -p chutoro-core --all-features --test result_api_surface '
            '--test session_api_surface $(BUILD_JOBS)',
            "",
            "expected three test Cargo invocations",
            id="missing-core-api-stable-run",
        ),
        pytest.param(
            '\tRUSTFLAGS="$(STABLE_VERIFY_RUSTFLAGS)" env -u RUSTUP_TOOLCHAIN '
            '$(CARGO) test -p chutoro-providers-dense --all-features '
            '--test portable_simd_gating $(BUILD_JOBS) -- --exact '
            'portable_simd_without_feature_is_rejected',
            "",
            "expected three test Cargo invocations",
            id="missing-dense-simd-stable-run",
        ),
        pytest.param(
            " & not (package(chutoro-core) & binary(result_api_surface))",
            "",
            "exclude exactly",
            id="missing-result-api-exclusion",
        ),
        pytest.param(
            " & not (package(chutoro-core) & binary(session_api_surface))",
            "",
            "exclude exactly",
            id="missing-session-api-exclusion",
        ),
        pytest.param(
            " & not (package(chutoro-providers-dense) & "
            "test(portable_simd_without_feature_is_rejected))",
            "",
            "exclude exactly",
            id="missing-simd-diagnostic-exclusion",
        ),
        pytest.param(
            "env -u RUSTUP_TOOLCHAIN $(CARGO) test -p chutoro-core",
            "env -u RUSTUP_TOOLCHAIN $(CARGO) --config $(DEV_FAST_CONFIG) "
            "test -p chutoro-core",
            "must not select the dev-fast fragment",
            id="stable-api-run-gains-dev-fast-fragment",
        ),
    ),
)
def test_test_boundary_assertion_rejects_routing_mutations(
    old: str, new: str, failure: str, tmp_path: Path
) -> None:
    """Each required stable run and exact nextest exclusion is binding."""
    makefile = _mutated_makefile(tmp_path, old, new)
    output = _make_output(
        "test", makefile=makefile, overrides=("DEV_FAST_HOST_OS=Linux",)
    )
    with pytest.raises(AssertionError, match=failure):
        _assert_test_boundary(output, "Linux")


@pytest.mark.parametrize(
    ("failing_package", "expected_invocations"),
    (
        pytest.param("chutoro-core", 2, id="core-api-verification"),
        pytest.param("chutoro-providers-dense", 3, id="dense-simd-verification"),
    ),
)
def test_make_test_propagates_a_stable_cargo_failure(
    failing_package: str, expected_invocations: int, tmp_path: Path
) -> None:
    """A failing stable verification leaf makes the Make gate fail."""
    invocations_path = tmp_path / "cargo-invocations.txt"
    fake_cargo = tmp_path / "fake-cargo.py"
    fake_cargo.write_text(
        f"""import sys
from pathlib import Path

with Path({str(invocations_path)!r}).open("a", encoding="utf-8") as output:
    output.write(repr(sys.argv[1:]) + "\\n")
if "nextest" in sys.argv:
    raise SystemExit(0)
if sys.argv[1:2] == ["test"] and {failing_package!r} in sys.argv:
    raise SystemExit(37)
raise SystemExit(0)
""",
        encoding="utf-8",
    )
    command = [
        "make",
        "--no-print-directory",
        "--file",
        str(MAKEFILE_PATH),
        "test",
        f"CARGO={sys.executable} {fake_cargo}",
        "DEV_FAST_HOST_OS=Darwin",
    ]
    completed = subprocess.run(
        command,
        cwd=MAKEFILE_PATH.parent,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0, (
        "make test must fail when its stable Cargo verification command fails"
    )
    invocations = invocations_path.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == expected_invocations, (
        f"expected nextest and stable runs through {failing_package}: {invocations!r}"
    )
    assert "nextest" in invocations[0], f"first leaf was not nextest: {invocations!r}"
    assert invocations[1].startswith("['test', '-p', 'chutoro-core'"), (
        f"the stable API surface verification did not run first: {invocations!r}"
    )
    if failing_package == "chutoro-providers-dense":
        assert invocations[2].startswith("['test', '-p', 'chutoro-providers-dense'"), (
            f"the dense SIMD verification did not fail after core: {invocations!r}"
        )
