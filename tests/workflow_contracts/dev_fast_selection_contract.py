"""The decision that `dev-test` selects the standard set minus one case.

The end-to-end suite can only make this decision after compiling the
workspace, which is why it is opt-in. The decision itself needs no build, so
it lives here and is exercised on every contract run against fabricated
selections. That places the omission this contract exists to catch — a
root-level case dropped from one side only — under the default suite.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

#: The case Cranelift cannot run. Written out rather than read from the
#: Makefile: a contract that imported the thing it checks could not notice a
#: rename on either side.
POISONED_LOCK_TEST = (
    "hnsw::cpu::test_helpers::tests::graph_helpers_report_poisoning_without_reconfiguring"
)


def selected(selection: set[str], test_name: str) -> set[str]:
    """Return the selection entries naming ``test_name``.

    Matching is by suffix, so an entry whose name has no module path — a
    root-level case — is compared like any other.

    >>> sorted(selected({"p root_case", "p suite nested::case"}, "root_case"))
    ['p root_case']
    """
    return {entry for entry in selection if entry.endswith(test_name)}


def assert_development_selection_is_standard_minus_one(
    dev_selection: set[str], standard_selection: set[str]
) -> None:
    """Require ``dev_selection`` to be the standard set minus one case.

    An entry is matched by suffix, so a root-level case — one whose name has no
    module path — is compared like any other. Dropping it from the development
    side alone must fail, even though such a name carries no ``::``.

    >>> standard = {
    ...     "p root_case",
    ...     f"chutoro-core {POISONED_LOCK_TEST}",
    ... }
    >>> assert_development_selection_is_standard_minus_one(
    ...     {"p root_case"}, standard
    ... ) is None
    True
    """
    assert selected(dev_selection, POISONED_LOCK_TEST) == set(), (
        "dev-test must not select the poisoned-lock regression"
    )
    assert len(selected(standard_selection, POISONED_LOCK_TEST)) == 1, (
        "the standard gate must select the poisoned-lock regression"
    )
    assert dev_selection == standard_selection - selected(
        standard_selection, POISONED_LOCK_TEST
    ), (
        "dev-test must select every standard case except the poisoned-lock "
        f"regression; unexpected difference: "
        f"{dev_selection ^ standard_selection}"
    )
