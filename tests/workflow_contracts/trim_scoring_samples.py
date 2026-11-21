"""Checked-in synthetic Rust sources that keep the trim-scoring reading honest.

Every real scorer on the fixed tree is serial, so a reading that scanned the
whole file, kept comments intact, or matched a bare substring would still pass
it. Each fixture under ``fixtures/trim_scoring`` is a shape the marker set has
to report, or has to leave alone. They are source files rather than inline
strings so a fixture change reads as Rust rather than as Python.

The historical scorer is a verbatim copy from commit
56d6909c7019ac7caa6d485d11ebf5708b47528e, ``chutoro-core/src/hnsw/cpu/trim.rs``
lines 35-48. It is checked in rather than read from Git history so the
contract keeps its evidence when history is shallow.

Re-use policy: import these from the trim-scoring tests only. A new sample
belongs in ``fixtures/trim_scoring`` with the shape it separates stated in
the file's comment, so a later reader knows which wrong reading it catches.
"""

from __future__ import annotations

import typing as typ
from pathlib import Path

FIXTURES: typ.Final[Path] = Path(__file__).with_name("fixtures") / "trim_scoring"


def fixture(name: str) -> str:
    """Return the Rust source of one checked-in fixture.

    Examples
    --------
    >>> "fn score_trim_jobs" in fixture("sequential")
    True
    """
    return (FIXTURES / f"{name}.rs").read_text(encoding="utf-8")

HISTORICAL_SCORE_TRIM_JOBS = fixture("historical_score_trim_jobs")

SEQUENTIAL = fixture("sequential")

DISPATCHING = fixture("dispatching")

COMMENTED_MENTION = fixture("commented_mention")

LITERAL_MENTION = fixture("literal_mention")

SLICE_JOIN = fixture("slice_join")

PATH_JOIN = fixture("path_join")

SPACED_RECEIVER_JOIN = fixture("spaced_receiver_join")

SLICE_JOIN_VARIABLE = fixture("slice_join_variable")

SLICE_JOIN_TRAILING_COMMA = fixture("slice_join_trailing_comma")

PATH_JOIN_TRAILING_COMMA = fixture("path_join_trailing_comma")

GENERIC_PATH_JOIN = fixture("generic_path_join")

NESTED_GENERIC_PATH_JOIN = fixture("nested_generic_path_join")

POOL_JOIN = fixture("pool_join")

GENERIC_FREE_JOIN = fixture("generic_free_join")

BOUND_JOIN = fixture("bound_join")

FACTORY_JOIN = fixture("factory_join")

UNIT_INSTALL = fixture("unit_install")

TUPLE_JOIN = fixture("tuple_join")

UNIT_TYPE_LOCAL = fixture("unit_type_local")

REVERSE_CHUNKS = fixture("reverse_chunks")

PAR_CHUNK_BY = fixture("par_chunk_by")

PAR_ARRAY_WINDOWS = fixture("par_array_windows")

PAR_CHAR_INDICES = fixture("par_char_indices")

PAR_BYTES = fixture("par_bytes")

PAR_ENCODE_UTF16 = fixture("par_encode_utf16")

PAR_MATCHES = fixture("par_matches")

PAR_MATCH_INDICES = fixture("par_match_indices")

CONST_EXPRESSION_INSTALL = fixture("const_expression_install")

QUALIFIED_PATH_JOIN = fixture("qualified_path_join")

BORROWED_QUALIFIED_PATH_JOIN = fixture("borrowed_qualified_path_join")

ASSOCIATED_SCOPE = fixture("associated_scope")

GENERIC_ASSOCIATED_INSTALL = fixture("generic_associated_install")

SCOPED_HELPER_NAME = fixture("scoped_helper_name")

UNLISTED_SIBLING = fixture("unlisted_sibling")

LISTED_HELPER_DISPATCH = fixture("listed_helper_dispatch")

RETURNED_FREE_JOIN = fixture("returned_free_join")

GENERIC_METHOD_SCOPE = fixture("generic_method_scope")

IN_PLACE_SCOPE = fixture("in_place_scope")

GROUPED_FREE_JOIN = fixture("grouped_free_join")

SPACED_TURBOFISH_INSTALL = fixture("spaced_turbofish_install")
