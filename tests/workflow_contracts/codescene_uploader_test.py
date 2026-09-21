"""Contract tests for the CodeScene uploader's deprecated checksum inputs.

At ``a5765019`` the shared uploader's committed ``cli-manifest.json`` is the
trust anchor for the CLI archive, and the action *rejects* a non-empty
``installer-checksum`` with a hard failure rather than ignoring it. A workflow
that still passes the input therefore breaks its upload step outright, and the
``CODESCENE_CLI_SHA256`` repository variable that fed it can only ever repeat
the manifest digest.

Four separate concerns are asserted, each in its own test so a failure names
the defect rather than a bundle:

* no workflow passes ``installer-checksum``;
* no workflow references the ``CODESCENE_CLI_SHA256`` variable that fed it;
* every uploader reference is pinned to one approved full SHA;
* the dispatch workflow that refreshed the variable is absent.

Each test asserts over a non-empty collection. A contract that ranges over an
empty collection is satisfied by deleting the thing it guards, so the
collections are checked for content before they are checked for compliance.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import re
import typing as typ

from workflow_support import ROOT, WORKFLOW_DIR, workflow_paths

#: The uploader revision this repository pins.
#:
#: ``a5765019`` is the floor the estate set for the manifest-verified
#: uploader, not a literal to pin. Dependabot had already carried this
#: repository past it to ``82feb2b7``, and
#: ``git diff a5765019 82feb2b7 -- .github/actions/upload-codescene-coverage/``
#: is empty, so the later pin carries the same action, the same committed
#: ``cli-manifest.json`` and the same rejection of ``installer-checksum``.
#: Naming the pin actually present keeps this an allowlist and avoids
#: downgrading a reference the estate has already moved forward.
UPLOADER_PIN: typ.Final = "82feb2b7aac45b7efff40c9c4bb632551b14521c"

#: Every ``uses:`` reference to the shared uploader, with its ref captured.
#: Matched against the file's raw text rather than a parsed document so that
#: a reference in a comment or a commented-out step is caught too.
UPLOADER_REFERENCE: typ.Final = re.compile(
    r"leynos/shared-actions/\.github/actions/upload-codescene-coverage@(\S+)"
)

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
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in paths
    }


def test_no_workflow_passes_the_deprecated_installer_checksum() -> None:
    """The uploader rejects a non-empty value, so no workflow may pass it."""
    offenders = sorted(
        name for name, text in _workflow_texts().items() if DEPRECATED_INPUT in text
    )
    assert not offenders, (
        f"{DEPRECATED_INPUT} is deprecated and rejected outright by the "
        f"uploader at {UPLOADER_PIN}; remove it from {', '.join(offenders)}"
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


def test_every_uploader_reference_is_pinned_to_the_approved_sha() -> None:
    """One approved SHA, asserted as an allowlist rather than as a floor.

    A floor would require ordering SHAs, which cannot be computed from a
    checkout. Naming the approved pin keeps the contract hermetic and fails
    closed on any other value, including a tag or a branch name.
    """
    references = {
        name: match.group(1)
        for name, text in _workflow_texts().items()
        for match in UPLOADER_REFERENCE.finditer(text)
    }
    assert references, (
        "no upload-codescene-coverage reference was found, so the pin "
        "assertion below would pass vacuously; this repository uploads "
        "coverage from main and checks it on pull requests"
    )
    wrong = {name: ref for name, ref in references.items() if ref != UPLOADER_PIN}
    assert not wrong, (
        "every upload-codescene-coverage reference must be pinned to "
        f"{UPLOADER_PIN}; found {wrong!r}"
    )


def test_the_checksum_refresh_workflow_is_absent() -> None:
    """Nothing consumes the variable it wrote, so the workflow is dead code."""
    refresh = WORKFLOW_DIR / REFRESH_WORKFLOW
    assert not refresh.exists(), (
        f"{REFRESH_WORKFLOW} refreshed {DEPRECATED_VARIABLE}, which no "
        "workflow reads any more; delete it rather than leave a dispatch that "
        "writes an unused repository variable"
    )
