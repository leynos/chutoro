"""Contract tests for the CodeScene uploader's deprecated checksum inputs.

From ``a5765019`` onward the shared uploader's committed
``cli-manifest.json`` is the trust anchor for the CLI archive, and the action
*rejects* a non-empty ``installer-checksum`` with a hard failure rather than
ignoring it. A workflow that still passes the input therefore breaks its
upload step outright, and the ``CODESCENE_CLI_SHA256`` repository variable
that fed it can only ever repeat the manifest digest.

Three separate concerns are asserted, each in its own test so a failure names
the defect rather than a bundle:

* no workflow passes ``installer-checksum``;
* no workflow references the ``CODESCENE_CLI_SHA256`` variable that fed it;
* the dispatch workflow that refreshed the variable is absent.

The uploader's pin is deliberately not asserted here. Dependabot owns its
value, and ``action_pins_test.py`` already requires every reference to be a
full commit SHA and the same SHA in every workflow; naming the value would
fail each routine bump, which the developers' guide forbids.

Each test asserts over a non-empty collection. A contract that ranges over an
empty collection is satisfied by deleting the thing it guards, so the
collections are checked for content before they are checked for compliance.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import typing as typ

from workflow_support import ROOT, WORKFLOW_DIR, read_workflow_text, workflow_paths

DEPRECATED_INPUT: typ.Final = "installer-checksum"
DEPRECATED_VARIABLE: typ.Final = "CODESCENE_CLI_SHA256"
REFRESH_WORKFLOW: typ.Final = "get-codescene-sha.yml"


def _workflow_texts() -> dict[str, str]:
    """Return every workflow's text, keyed by repository-relative path.

    Returns
    -------
    dict[str, str]
        One entry per workflow file, in sorted order so a failure lists the
        offenders predictably.
    """
    paths = workflow_paths()
    assert paths, (
        f"no workflow files were found under {WORKFLOW_DIR}, so every "
        "contract below would pass having read nothing"
    )
    return {
        path.relative_to(ROOT).as_posix(): read_workflow_text(path) for path in paths
    }


def test_no_workflow_passes_the_deprecated_installer_checksum() -> None:
    """The uploader rejects a non-empty value, so no workflow may pass it."""
    offenders = sorted(
        name for name, text in _workflow_texts().items() if DEPRECATED_INPUT in text
    )
    assert not offenders, (
        f"{DEPRECATED_INPUT} is deprecated and rejected outright by the "
        f"manifest-verified uploader; remove it from {', '.join(offenders)}"
    )


def test_no_workflow_references_the_deprecated_checksum_variable() -> None:
    """The variable existed only to feed the rejected input, so it must go."""
    offenders = sorted(
        name for name, text in _workflow_texts().items() if DEPRECATED_VARIABLE in text
    )
    assert not offenders, (
        f"{DEPRECATED_VARIABLE} fed the deprecated installer checksum and has "
        f"no remaining consumer; remove it from {', '.join(offenders)}"
    )


def test_the_checksum_refresh_workflow_is_absent() -> None:
    """Nothing consumes the variable it wrote, so the workflow is dead code."""
    refresh = WORKFLOW_DIR / REFRESH_WORKFLOW
    assert not refresh.exists(), (
        f"{REFRESH_WORKFLOW} refreshed {DEPRECATED_VARIABLE}, which no "
        "workflow reads any more; delete it rather than leave a dispatch that "
        "writes an unused repository variable"
    )
