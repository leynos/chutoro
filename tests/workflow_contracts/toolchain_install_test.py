"""Contract-test how workflows invoke rustup.

`rustup --component` takes one value. Written as `--component clippy
rustfmt`, the second name is parsed as a toolchain and rustup exits 1, so
the lane dies before it does any work. `nightly-portable-simd` shipped
that for as long as it has existed and never once ran its tests: on a day
its gate fired it failed here, and on every other day the gate skipped the
work and the run reported success in seconds. Nothing in the repository
could see the difference, because the flag is only wrong at the point
rustup parses it.

These tests read the invocation rather than the step's name, so deleting
the fix while keeping the step fails them. The reading itself, and the
nightly lane's presence contract over it, live in `rustup_install_reading.py`
so this module stays within the 400-line limit.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import tomllib
import typing as typ

import pytest
from rustup_install_reading import (
    COMPONENT_FLAG,
    join_continuations,
    NIGHTLY_JOB,
    NIGHTLY_WORKFLOW,
    assert_nightly_installer,
    component_groups,
    rustup_installs,
    rustup_invocations,
    run_scripts,
)
from workflow_support import (
    ROOT,
    job,
    jobs,
    workflow_names,
)


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_each_rustup_component_carries_its_own_flag(workflow_name: str) -> None:
    """Fail when one `--component` flag is given more than one value.

    rustup reads the second value as a toolchain name and exits before
    installing anything, so the step fails and every step after it is
    skipped.
    """
    for group in component_groups(workflow_name):
        assert len(group.values) == 1, (
            f"{workflow_name}:{group.job_name} installs {group.values} under "
            f"one {COMPONENT_FLAG} flag; rustup reads {group.values[1:]} as "
            "toolchain names and exits 1"
        )


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_no_rustup_install_asks_for_a_component_it_does_not_name(
    workflow_name: str,
) -> None:
    """Every `--component` flag has a value, rather than swallowing the next flag."""
    for group in component_groups(workflow_name):
        assert group.values, (
            f"{workflow_name}:{group.job_name} passes {COMPONENT_FLAG} with no "
            f"value in {group.invocation!r}"
        )


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_no_rustup_install_ends_on_a_dangling_backslash(workflow_name: str) -> None:
    """A rustup install must not be truncated by a broken continuation.

    `\\` followed by a space and a newline is not a continuation: the
    backslash escapes the space and the shell ends the command at the newline,
    dropping every flag on the lines below. The run then installs a toolchain
    with none of the components asked for, and succeeds, so nothing downstream
    reports it. The reading above leaves such a backslash in place precisely
    so it can be seen here.
    """
    for job_name, invocation in rustup_invocations(workflow_name):
        assert not invocation.rstrip().endswith("\\"), (
            f"{workflow_name}:{job_name} ends a rustup install on a backslash "
            f"the shell does not read as a continuation, so everything after "
            f"it is dropped: {invocation!r}"
        )


def test_the_nightly_lane_installs_nightly_with_clippy_and_rustfmt() -> None:
    """Selecting the installer by its `run` content keeps the contract real.

    A workflow that dropped the installer would otherwise pass a contract
    written as "every installer names both components", because the loop
    over an empty selection proves nothing. The assertion therefore requires
    the one invocation before it reads anything from it.
    """
    assert_nightly_installer(job(NIGHTLY_WORKFLOW, NIGHTLY_JOB))


#: The stage of the nightly lane whose installer the injected cases vary.
_INSTALLER = "rustup toolchain install nightly --profile minimal"

#: A compile-contract step, so the no-installer case still describes a lane
#: that would try to run its tests were the installer present.
_CONTRACT_STEP = {
    "name": "Test portable SIMD compile contract",
    "run": (
        "cargo +nightly test -p chutoro-compile-contracts "
        "--test portable_simd_gating"
    ),
}


def _nightly_workflow(installer: str | None) -> dict[str, typ.Any]:
    """Return a one-job workflow carrying *installer*, or none when ``None``."""
    install_steps = [] if installer is None else [{"run": installer}]
    return {
        "jobs": {
            NIGHTLY_JOB: {
                "steps": [
                    {
                        "uses": (
                            "actions/checkout@"
                            "3d3c42e5aac5ba805825da76410c181273ba90b1"
                        )
                    },
                    *install_steps,
                    _CONTRACT_STEP,
                ]
            }
        }
    }


@pytest.mark.parametrize(
    ("workflow", "is_accepted"),
    (
        pytest.param(
            _nightly_workflow(
                f"{_INSTALLER} --component clippy --component rustfmt"
            ),
            True,
            id="repeated-flags",
        ),
        pytest.param(
            _nightly_workflow(f"{_INSTALLER} --component clippy,rustfmt"),
            True,
            id="comma-list",
        ),
        pytest.param(
            _nightly_workflow(
                'rustup toolchain install "nightly" --profile "minimal" '
                '--component "clippy,rustfmt"'
            ),
            True,
            id="quoted-values",
        ),
        pytest.param(
            _nightly_workflow(
                f"{_INSTALLER} \\\n  --component clippy --component rustfmt"
            ),
            True,
            id="continued-repeated-flags",
        ),
        pytest.param(
            _nightly_workflow(
                "echo 'rustup toolchain install nightly --profile minimal "
                "--component clippy --component rustfmt'"
            ),
            False,
            id="quoted-echo-is-not-an-installer",
        ),
        pytest.param(
            _nightly_workflow(
                "printf '%s\\n' \"setup\n"
                "rustup toolchain install nightly --profile minimal "
                "--component clippy,rustfmt\n\""
            ),
            False,
            id="quoted-multiline-installer-is-not-a-command",
        ),
        pytest.param(
            _nightly_workflow(
                "echo ready;# disabled; rustup toolchain install nightly "
                "--profile minimal --component clippy --component rustfmt"
            ),
            False,
            id="comment-after-semicolon",
        ),
        pytest.param(
            _nightly_workflow(
                "rustup toolchain install nightly --profile minimal \\\\\n"
                "  --component clippy --component rustfmt"
            ),
            False,
            id="escaped-backslash-is-not-a-continuation",
        ),
        pytest.param(
            _nightly_workflow(
                "echo \\; rustup toolchain install nightly --profile minimal "
                "--component clippy --component rustfmt"
            ),
            False,
            id="escaped-semicolon-is-not-a-boundary",
        ),
        pytest.param(
            _nightly_workflow(
                "rustup toolchain\ninstall nightly --profile minimal "
                "--component clippy --component rustfmt"
            ),
            False,
            id="unescaped-newline-between-command-words",
        ),
        pytest.param(
            _nightly_workflow(
                '"echo" rustup toolchain install nightly --profile minimal '
                "--component clippy --component rustfmt"
            ),
            False,
            id="quoted-command-word",
        ),
        pytest.param(
            _nightly_workflow(
                "rustup toolchain install-extra nightly --profile minimal "
                "--component clippy --component rustfmt"
            ),
            False,
            id="install-extra-is-not-install",
        ),
        pytest.param(
            _nightly_workflow(
                '"rustup" toolchain install nightly --profile minimal '
                "--component clippy --component rustfmt"
            ),
            True,
            id="quoted-executable-name",
        ),
        pytest.param(
            _nightly_workflow(
                "cat <<'EOF'\n"
                "rustup toolchain install nightly --profile minimal "
                "--component clippy --component rustfmt\n"
                "EOF"
            ),
            False,
            id="here-document-body-is-not-a-command",
        ),
        pytest.param(
            _nightly_workflow(
                "cat <<< ready\n"
                "rustup toolchain install nightly --profile minimal "
                "--component clippy --component rustfmt"
            ),
            True,
            id="installer-after-here-string",
        ),
        pytest.param(
            _nightly_workflow(
                "RUSTUP_TOOLCHAIN=stable rustup toolchain install nightly "
                "--profile minimal --component clippy --component rustfmt"
            ),
            True,
            id="assignment-prefixed-installer",
        ),
        pytest.param(
            _nightly_workflow(
                'echo \\"; rustup toolchain install nightly --profile minimal '
                '--component clippy --component rustfmt; echo "done"'
            ),
            True,
            id="escaped-quote-does-not-open-a-span",
        ),
        pytest.param(
            _nightly_workflow(
                "cat <<EOF\n EOF\n"
                "rustup toolchain install nightly --profile minimal "
                "--component clippy --component rustfmt\nEOF"
            ),
            False,
            id="indented-delimiter-does-not-close",
        ),
        pytest.param(
            _nightly_workflow(
                "cat <<'EOF'\ndon't\nEOF\n"
                "rustup toolchain install nightly --profile minimal "
                "--component clippy,rustfmt # required components"
            ),
            True,
            id="here-document-body-does-not-change-quote-state",
        ),
        pytest.param(
            _nightly_workflow(
                'LABEL="nightly setup" rustup toolchain install nightly '
                "--profile minimal --component clippy --component rustfmt"
            ),
            True,
            id="quoted-assignment-value-with-space",
        ),
        pytest.param(
            _nightly_workflow(
                'printf "%s" "setup\n<<EOF\n"\n'
                "rustup toolchain install nightly --profile minimal "
                "--component clippy --component rustfmt"
            ),
            True,
            id="quoted-multiline-here-document-operator",
        ),
        pytest.param(
            _nightly_workflow(
                "echo \\ # setup; rustup toolchain install nightly "
                "--profile minimal --component clippy --component rustfmt"
            ),
            True,
            id="escaped-space-does-not-start-a-comment",
        ),
        pytest.param(
            _nightly_workflow(
                "if rustup toolchain install nightly --profile minimal "
                "--component clippy,rustfmt; then :; fi"
            ),
            True,
            id="installer-in-a-shell-condition",
        ),
        pytest.param(_nightly_workflow(None), False, id="no-installer"),
        pytest.param(
            _nightly_workflow(
                f"# {_INSTALLER} --component clippy --component rustfmt"
            ),
            False,
            id="commented-out-installer",
        ),
        pytest.param(
            _nightly_workflow(f"{_INSTALLER} --component clippy"),
            False,
            id="missing-rustfmt",
        ),
        pytest.param(
            _nightly_workflow(
                "rustup toolchain install stable --profile minimal "
                "--component clippy --component rustfmt"
            ),
            False,
            id="wrong-toolchain",
        ),
        pytest.param(
            _nightly_workflow(
                "rustup toolchain install nightly --profile default "
                "--component clippy --component rustfmt"
            ),
            False,
            id="wrong-profile",
        ),
    ),
)
def test_the_nightly_installer_contract_rejects_only_broken_installers(
    workflow: dict[str, typ.Any],
    is_accepted: bool,
) -> None:
    """The presence, toolchain, profile, and component checks must all bite.

    Each case injects a parsed workflow directly, so the assertion is
    exercised without editing the real lane or mutating the module's loader.
    The malformed single-flag command is absent here on purpose:
    `test_each_rustup_component_carries_its_own_flag` already rejects it for
    every workflow, and repeating it would test one defect twice.
    """
    job_definition = jobs(workflow)[NIGHTLY_JOB]
    if is_accepted:
        assert_nightly_installer(job_definition)
    else:
        with pytest.raises(AssertionError):
            assert_nightly_installer(job_definition)


def test_nightly_lane_runs_portable_simd_compile_contract() -> None:
    """Nightly must execute the feature-enabled portable-SIMD contract."""
    commands = [
        " ".join(join_continuations(script).split())
        for _, script in run_scripts("nightly-portable-simd.yml")
        if "cargo +nightly test" in script
    ]
    assert (
        "cargo fetch --locked && RUSTUP_TOOLCHAIN=nightly cargo +nightly test "
        "-p chutoro-compile-contracts --features nightly_portable_simd "
        "--test portable_simd_gating" in commands
    ), (
        "nightly lane must fetch the locked dependencies, then run "
        "RUSTUP_TOOLCHAIN=nightly cargo +nightly test "
        "-p chutoro-compile-contracts --features nightly_portable_simd "
        "--test portable_simd_gating"
    )


def test_nightly_lane_runs_portable_simd_contract_without_feature() -> None:
    """Nightly also checks the compile refusal before enabling the feature."""
    commands = [
        " ".join(join_continuations(script).split())
        for _, script in run_scripts("nightly-portable-simd.yml")
        if "cargo +nightly test" in script
    ]
    assert (
        "cargo fetch --locked && RUSTUP_TOOLCHAIN=nightly cargo +nightly test "
        "-p chutoro-compile-contracts --test portable_simd_gating" in commands
    ), "nightly lane must run the portable SIMD compile contract without its feature"


def test_portable_simd_feature_forwards_to_the_dense_provider() -> None:
    """The compile-contract feature resolves the provider's public feature edge."""
    contracts = tomllib.loads(
        (ROOT / "chutoro-compile-contracts/Cargo.toml").read_text(encoding="utf-8")
    )
    provider = tomllib.loads(
        (ROOT / "chutoro-providers/dense/Cargo.toml").read_text(encoding="utf-8")
    )
    assert contracts["features"]["nightly_portable_simd"] == [
        "chutoro-providers-dense/nightly_portable_simd"
    ], "compile-contract opt-in must forward the dense provider feature"
    assert (
        contracts["dev-dependencies"]["chutoro-providers-dense"]["path"]
        == "../chutoro-providers/dense"
    ), "the feature edge must resolve the workspace dense provider dependency"
    assert "nightly_portable_simd" in provider["features"], (
        "the forwarded feature must exist in the dense provider manifest"
    )


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        pytest.param(
            "rustup toolchain install nightly \\\n  --component clippy",
            ["rustup toolchain install nightly   --component clippy"],
            id="continuation-is-joined",
        ),
        pytest.param(
            "rustup toolchain install nightly --profile minimal "
            "--component clippy,\\\nrustfmt",
            [
                "rustup toolchain install nightly --profile minimal "
                "--component clippy,rustfmt"
            ],
            id="continuation-inside-an-argument",
        ),
        pytest.param(
            "printf '%s\\n' \"setup\n"
            "rustup toolchain install nightly --profile minimal "
            "--component clippy,rustfmt\n\"",
            [],
            id="quoted-multiline-installer-is-not-a-command",
        ),
        pytest.param(
            "echo ready;# disabled; rustup toolchain install nightly",
            [],
            id="comment-after-semicolon",
        ),
        pytest.param(
            "rustup toolchain install nightly --profile minimal \\\\\n"
            "  --component clippy",
            ["rustup toolchain install nightly --profile minimal \\\\"],
            id="escaped-backslash-ends-the-command",
        ),
        pytest.param(
            "echo \\; rustup toolchain install nightly",
            [],
            id="escaped-semicolon-is-not-a-boundary",
        ),
        pytest.param(
            "rustup toolchain\ninstall nightly --profile minimal",
            [],
            id="unescaped-newline-between-command-words",
        ),
        pytest.param(
            '"echo" rustup toolchain install nightly',
            [],
            id="quoted-command-word",
        ),
        pytest.param(
            "rustup toolchain install-extra nightly --profile minimal",
            [],
            id="install-extra-is-not-install",
        ),
        pytest.param(
            '"rustup" toolchain install nightly --profile minimal',
            ['"rustup" toolchain install nightly --profile minimal'],
            id="quoted-executable-name",
        ),
        pytest.param(
            "cat <<'EOF'\n"
            "rustup toolchain install nightly --profile minimal\n"
            "EOF",
            [],
            id="here-document-body-is-not-a-command",
        ),
        pytest.param(
            "cat <<EOF\nbody\nEOF\n"
            "rustup toolchain install nightly --profile minimal",
            ["rustup toolchain install nightly --profile minimal"],
            id="installer-after-here-document",
        ),
        pytest.param(
            "cat <<< ready\nrustup toolchain install nightly",
            ["rustup toolchain install nightly"],
            id="installer-after-here-string",
        ),
        pytest.param(
            "RUSTUP_TOOLCHAIN=stable rustup toolchain install nightly "
            "--profile minimal",
            ["rustup toolchain install nightly --profile minimal"],
            id="assignment-prefixed-installer",
        ),
        pytest.param(
            'echo \\"; rustup toolchain install nightly; echo "done"',
            ["rustup toolchain install nightly"],
            id="escaped-quote-does-not-open-a-span",
        ),
        pytest.param(
            "cat <<EOF\n EOF\nrustup toolchain install nightly\nEOF",
            [],
            id="indented-delimiter-does-not-close",
        ),
        pytest.param(
            'echo "<<EOF"\n'
            "rustup toolchain install nightly --profile minimal "
            "--component clippy rustfmt",
            [
                "rustup toolchain install nightly --profile minimal "
                "--component clippy rustfmt"
            ],
            id="printed-operator-does-not-hide-installer",
        ),
        pytest.param(
            "cat <<'EOF'\ndon't\nEOF\n"
            "rustup toolchain install nightly --profile minimal "
            "--component clippy,rustfmt # required components",
            [
                "rustup toolchain install nightly --profile minimal "
                "--component clippy,rustfmt "
            ],
            id="here-document-body-does-not-change-quote-state",
        ),
        pytest.param(
            'LABEL="nightly setup" rustup toolchain install nightly '
            "--profile minimal",
            ["rustup toolchain install nightly --profile minimal"],
            id="quoted-assignment-value-with-space",
        ),
        pytest.param(
            'printf "%s" "setup\n<<EOF\n"\n'
            "rustup toolchain install nightly --profile minimal",
            ["rustup toolchain install nightly --profile minimal"],
            id="quoted-multiline-here-document-operator",
        ),
        pytest.param(
            "echo \\ # setup; rustup toolchain install nightly",
            ["rustup toolchain install nightly"],
            id="escaped-space-does-not-start-a-comment",
        ),
        pytest.param(
            "if rustup toolchain install nightly --profile minimal "
            "--component clippy,rustfmt; then :; fi",
            [
                "rustup toolchain install nightly --profile minimal "
                "--component clippy,rustfmt"
            ],
            id="installer-in-a-shell-condition",
        ),
        pytest.param(
            "rustup toolchain install nightly \\ \n  --component clippy",
            ["rustup toolchain install nightly \\ "],
            id="backslash-space-newline-is-not-a-continuation",
        ),
        pytest.param(
            "rustup toolchain install nightly \\\t\n  --component clippy",
            ["rustup toolchain install nightly \\\t"],
            id="backslash-tab-newline-is-not-a-continuation",
        ),
        pytest.param(
            "printf '%s\\n' \"setup\n# setup complete\"; "
            "rustup toolchain install nightly --profile minimal "
            "--component clippy",
            [
                "rustup toolchain install nightly --profile minimal "
                "--component clippy"
            ],
            id="quote-state-survives-a-line-break",
        ),
    ],
)
def test_the_reading_joins_only_what_the_shell_joins(
    script: str, expected: list[str]
) -> None:
    """Whitespace between a backslash and a newline ends the command.

    Joining it anyway would let a truncated install read as a well-formed one,
    which is the defect this module exists to catch rather than to hide.
    """
    assert rustup_installs(script) == expected
