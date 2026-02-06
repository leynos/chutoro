//! Ordering proofs for edge harvest primitives.

use vstd::prelude::*;
use vstd::relations::{antisymmetric, reflexive, strongly_connected, total_ordering, transitive};

verus! {

use super::*;

/// Proves that `edge_leq` is a total ordering for candidate edges.
///
/// ## Example
/// ```text
/// After calling this lemma, sorting by `edge_leq` is permitted in proofs.
/// ```
pub(super) proof fn lemma_edge_leq_total_ordering()
    ensures
        total_ordering(|a: CandidateEdgeSpec, b: CandidateEdgeSpec| edge_leq(a, b)),
{
    assert(total_ordering(|a: CandidateEdgeSpec, b: CandidateEdgeSpec| edge_ord_leq(a, b))) by {
        reveal(total_ordering);

        assert(reflexive(|a: CandidateEdgeSpec, b: CandidateEdgeSpec| edge_ord_leq(a, b))) by {
            assert forall|a: CandidateEdgeSpec| #![auto] edge_ord_leq(a, a) by {
                assert(edge_ord_leq(a, a));
            }
        }

        assert(antisymmetric(|a: CandidateEdgeSpec, b: CandidateEdgeSpec| edge_ord_leq(a, b))) by {
            assert forall|a: CandidateEdgeSpec, b: CandidateEdgeSpec|
                #![auto]
                edge_ord_leq(a, b) && edge_ord_leq(b, a) implies a == b by {
                if a.distance > b.distance {
                    assert(edge_ord_leq(a, b) == false);
                }
                if b.distance > a.distance {
                    assert(edge_ord_leq(b, a) == false);
                }
                assert(a.distance == b.distance);

                if a.source > b.source {
                    assert(edge_ord_leq(a, b) == false);
                }
                if b.source > a.source {
                    assert(edge_ord_leq(b, a) == false);
                }
                assert(a.source == b.source);

                if a.target > b.target {
                    assert(edge_ord_leq(a, b) == false);
                }
                if b.target > a.target {
                    assert(edge_ord_leq(b, a) == false);
                }
                assert(a.target == b.target);

                if a.sequence > b.sequence {
                    assert(edge_ord_leq(a, b) == false);
                }
                if b.sequence > a.sequence {
                    assert(edge_ord_leq(b, a) == false);
                }
                assert(a.sequence == b.sequence);
                assert(a == b);
            }
        }

        assert(transitive(|a: CandidateEdgeSpec, b: CandidateEdgeSpec| edge_ord_leq(a, b))) by {
            assert forall|a: CandidateEdgeSpec, b: CandidateEdgeSpec, c: CandidateEdgeSpec|
                #![auto]
                edge_ord_leq(a, b) && edge_ord_leq(b, c) implies edge_ord_leq(a, c) by {
                if a.distance < b.distance {
                    if b.distance > c.distance {
                        assert(edge_ord_leq(b, c) == false);
                    }
                    assert(a.distance < c.distance);
                    assert(edge_ord_leq(a, c));
                } else if a.distance > b.distance {
                    assert(edge_ord_leq(a, b) == false);
                } else if b.distance < c.distance {
                    assert(a.distance < c.distance);
                    assert(edge_ord_leq(a, c));
                } else if b.distance > c.distance {
                    assert(edge_ord_leq(b, c) == false);
                } else {
                    if a.source < b.source {
                        if b.source > c.source {
                            assert(edge_ord_leq(b, c) == false);
                        }
                        assert(a.source < c.source);
                        assert(edge_ord_leq(a, c));
                    } else if a.source > b.source {
                        assert(edge_ord_leq(a, b) == false);
                    } else if b.source < c.source {
                        assert(a.source < c.source);
                        assert(edge_ord_leq(a, c));
                    } else if b.source > c.source {
                        assert(edge_ord_leq(b, c) == false);
                    } else {
                        if a.target < b.target {
                            if b.target > c.target {
                                assert(edge_ord_leq(b, c) == false);
                            }
                            assert(a.target < c.target);
                            assert(edge_ord_leq(a, c));
                        } else if a.target > b.target {
                            assert(edge_ord_leq(a, b) == false);
                        } else if b.target < c.target {
                            assert(a.target < c.target);
                            assert(edge_ord_leq(a, c));
                        } else if b.target > c.target {
                            assert(edge_ord_leq(b, c) == false);
                        } else {
                            if a.sequence <= b.sequence {
                                if b.sequence <= c.sequence {
                                    assert(edge_ord_leq(a, c));
                                } else {
                                    assert(edge_ord_leq(b, c) == false);
                                }
                            } else {
                                assert(edge_ord_leq(a, b) == false);
                            }
                        }
                    }
                }
            }
        }

        assert(strongly_connected(|a: CandidateEdgeSpec, b: CandidateEdgeSpec| edge_ord_leq(a, b))) by {
            assert forall|a: CandidateEdgeSpec, b: CandidateEdgeSpec|
                #![auto]
                edge_ord_leq(a, b) || edge_ord_leq(b, a) by {
                if a.distance < b.distance {
                    assert(edge_ord_leq(a, b));
                } else if a.distance > b.distance {
                    assert(edge_ord_leq(b, a));
                } else if a.source < b.source {
                    assert(edge_ord_leq(a, b));
                } else if a.source > b.source {
                    assert(edge_ord_leq(b, a));
                } else if a.target < b.target {
                    assert(edge_ord_leq(a, b));
                } else if a.target > b.target {
                    assert(edge_ord_leq(b, a));
                } else if a.sequence <= b.sequence {
                    assert(edge_ord_leq(a, b));
                } else {
                    assert(edge_ord_leq(b, a));
                }
            }
        }
    }

    assert(total_ordering(|a: CandidateEdgeSpec, b: CandidateEdgeSpec| edge_leq(a, b))) by {
        reveal(total_ordering);

        assert(reflexive(|a: CandidateEdgeSpec, b: CandidateEdgeSpec| edge_leq(a, b))) by {
            assert forall|a: CandidateEdgeSpec| #![auto] edge_leq(a, a) by {
                assert(edge_ord_leq(a, a));
            }
        }

        assert(antisymmetric(|a: CandidateEdgeSpec, b: CandidateEdgeSpec| edge_leq(a, b))) by {
            assert forall|a: CandidateEdgeSpec, b: CandidateEdgeSpec|
                #![auto]
                edge_leq(a, b) && edge_leq(b, a) implies a == b by {
                if a.sequence < b.sequence {
                    assert(edge_leq(b, a) == false);
                } else if a.sequence > b.sequence {
                    assert(edge_leq(a, b) == false);
                } else {
                    assert(edge_ord_leq(a, b) && edge_ord_leq(b, a));
                    assert(a == b) by {
                        assert(antisymmetric(|a: CandidateEdgeSpec, b: CandidateEdgeSpec| edge_ord_leq(a, b)));
                    }
                }
            }
        }

        assert(transitive(|a: CandidateEdgeSpec, b: CandidateEdgeSpec| edge_leq(a, b))) by {
            assert forall|a: CandidateEdgeSpec, b: CandidateEdgeSpec, c: CandidateEdgeSpec|
                #![auto]
                edge_leq(a, b) && edge_leq(b, c) implies edge_leq(a, c) by {
                if a.sequence < b.sequence {
                    if b.sequence > c.sequence {
                        assert(edge_leq(b, c) == false);
                    }
                    assert(a.sequence < c.sequence);
                    assert(edge_leq(a, c));
                } else if a.sequence > b.sequence {
                    assert(edge_leq(a, b) == false);
                } else if b.sequence < c.sequence {
                    assert(a.sequence < c.sequence);
                    assert(edge_leq(a, c));
                } else if b.sequence > c.sequence {
                    assert(edge_leq(b, c) == false);
                } else {
                    assert(edge_ord_leq(a, b) && edge_ord_leq(b, c));
                    assert(edge_ord_leq(a, c)) by {
                        assert(transitive(|a: CandidateEdgeSpec, b: CandidateEdgeSpec| edge_ord_leq(a, b)));
                    }
                    assert(edge_leq(a, c));
                }
            }
        }

        assert(strongly_connected(|a: CandidateEdgeSpec, b: CandidateEdgeSpec| edge_leq(a, b))) by {
            assert forall|a: CandidateEdgeSpec, b: CandidateEdgeSpec|
                #![auto]
                edge_leq(a, b) || edge_leq(b, a) by {
                if a.sequence < b.sequence {
                    assert(edge_leq(a, b));
                } else if a.sequence > b.sequence {
                    assert(edge_leq(b, a));
                } else {
                    assert(edge_ord_leq(a, b) || edge_ord_leq(b, a));
                }
            }
        }
    }
}

/// Proves that endpoint ordering preserves fields.
///
/// ## Example
/// ```text
/// Edges with source > target yield swapped endpoints.
/// ```
pub(super) proof fn lemma_canonicalise_preserves_fields(edge: CandidateEdgeSpec)
    ensures
        edge.canonicalise().distance == edge.distance,
        edge.canonicalise().sequence == edge.sequence,
        edge.canonicalise().source <= edge.canonicalise().target,
        edge.canonicalise().source
            == if edge.source <= edge.target {
                edge.source
            } else {
                edge.target
            },
        edge.canonicalise().target
            == if edge.source <= edge.target {
                edge.target
            } else {
                edge.source
            },
{
    let canonical = edge.canonicalise();

    if edge.source <= edge.target {
        assert(canonical == edge);
        assert(canonical.source <= canonical.target);
    } else {
        assert(canonical.source == edge.target);
        assert(canonical.target == edge.source);
        assert(canonical.distance == edge.distance);
        assert(canonical.sequence == edge.sequence);
        assert(canonical.source <= canonical.target);
    }
}

} // verus!
