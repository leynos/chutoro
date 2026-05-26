//! Kani proof harnesses for dense SIMD boundary contracts.
//!
//! These harnesses keep the low-level SIMD safety policy executable without
//! modelling raw architecture intrinsics. They prove the support-mask selector
//! and the bounded lane arithmetic used by packed query-to-points kernels.

use super::dispatch::{self, CompiledSimdSupport, RuntimeSimdSupport};
use super::lane_output_count;
use super::point_view;

const MAX_PROOF_POINTS: usize = 17;
const MAX_PROOF_DIMENSION: usize = 3;
const MAX_PROOF_BATCHES: usize = 8;

#[kani::proof]
fn verify_dense_simd_dispatch_selection_respects_support_masks() {
    let avx512 = SimdBackendFlag {
        compiled: kani::any(),
        runtime: kani::any(),
    };
    let avx2 = SimdBackendFlag {
        compiled: kani::any(),
        runtime: kani::any(),
    };
    let neon = SimdBackendFlag {
        compiled: kani::any(),
        runtime: kani::any(),
    };
    let psimd = SimdBackendFlag {
        compiled: kani::any(),
        runtime: kani::any(),
    };

    let compiled = CompiledSimdSupport::new(
        avx2.compiled,
        avx512.compiled,
        neon.compiled,
        psimd.compiled,
    );
    let runtime =
        RuntimeSimdSupport::new(avx2.runtime, avx512.runtime, neon.runtime, psimd.runtime);

    match dispatch::choose_euclidean_backend(compiled, runtime) {
        dispatch::EuclideanBackend::Avx512 => {
            assert_eligible(avx512, EligibleBackend::Avx512);
        }
        dispatch::EuclideanBackend::Avx2 => {
            assert_not_eligible(avx512, IneligibleBackend::Avx512Priority);
            assert_eligible(avx2, EligibleBackend::Avx2);
        }
        dispatch::EuclideanBackend::Neon => {
            assert_not_eligible(avx512, IneligibleBackend::Avx512Priority);
            assert_not_eligible(avx2, IneligibleBackend::Avx2Priority);
            assert_eligible(neon, EligibleBackend::Neon);
        }
        dispatch::EuclideanBackend::PortableSimd => {
            assert_not_eligible(avx512, IneligibleBackend::Avx512Priority);
            assert_not_eligible(avx2, IneligibleBackend::Avx2Priority);
            assert_not_eligible(neon, IneligibleBackend::NeonPriority);
            assert_eligible(psimd, EligibleBackend::PortableSimd);
        }
        dispatch::EuclideanBackend::Scalar => {
            assert_not_eligible(avx512, IneligibleBackend::Avx512Absent);
            assert_not_eligible(avx2, IneligibleBackend::Avx2Absent);
            assert_not_eligible(neon, IneligibleBackend::NeonAbsent);
            assert_not_eligible(psimd, IneligibleBackend::PortableSimdAbsent);
        }
    }
}

#[kani::proof]
#[kani::unwind(20)]
fn verify_dense_simd_tail_padding_lane_bounds() {
    let point_count = bounded_usize(MAX_PROOF_POINTS);
    let dimension = bounded_usize(MAX_PROOF_DIMENSION);
    let lanes = symbolic_lane_width();
    let padded_count = point_view::padded_point_count(point_count);

    kani::assert(
        padded_count % super::MAX_SIMD_LANES == 0,
        "padded point count must be a 16-lane multiple",
    );
    kani::assert(
        padded_count >= point_count,
        "padded point count must cover logical points",
    );

    let dimension_index = bounded_usize(MAX_PROOF_DIMENSION);
    kani::assume(dimension == 0 || dimension_index < dimension);
    verify_lane_loads(point_count, padded_count, lanes);
}

/// Represents whether a SIMD backend is both compiled-in and available at runtime.
#[derive(Clone, Copy)]
struct SimdBackendFlag {
    compiled: bool,
    runtime: bool,
}

impl SimdBackendFlag {
    fn eligible(&self) -> bool {
        self.compiled && self.runtime
    }
}

enum IneligibleBackend {
    Avx512Priority,
    Avx2Priority,
    NeonPriority,
    Avx512Absent,
    Avx2Absent,
    NeonAbsent,
    PortableSimdAbsent,
}

enum EligibleBackend {
    Avx512,
    Avx2,
    Neon,
    PortableSimd,
}

fn assert_not_eligible(flag: SimdBackendFlag, backend: IneligibleBackend) {
    match backend {
        IneligibleBackend::Avx512Priority => {
            kani::assert(!flag.eligible(), "AVX-512 has priority");
        }
        IneligibleBackend::Avx2Priority => {
            kani::assert(!flag.eligible(), "AVX2 has priority");
        }
        IneligibleBackend::NeonPriority => {
            kani::assert(!flag.eligible(), "NEON has priority");
        }
        IneligibleBackend::Avx512Absent => {
            kani::assert(!flag.eligible(), "AVX-512 must be absent");
        }
        IneligibleBackend::Avx2Absent => {
            kani::assert(!flag.eligible(), "AVX2 must be absent");
        }
        IneligibleBackend::NeonAbsent => {
            kani::assert(!flag.eligible(), "NEON must be absent");
        }
        IneligibleBackend::PortableSimdAbsent => {
            kani::assert(!flag.eligible(), "portable SIMD must be absent");
        }
    }
}

fn assert_eligible(flag: SimdBackendFlag, backend: EligibleBackend) {
    match backend {
        EligibleBackend::Avx512 => {
            kani::assert(flag.compiled, "AVX-512 must be compiled");
            kani::assert(flag.runtime, "AVX-512 must be available");
        }
        EligibleBackend::Avx2 => {
            kani::assert(flag.compiled, "AVX2 must be compiled");
            kani::assert(flag.runtime, "AVX2 must be available");
        }
        EligibleBackend::Neon => {
            kani::assert(flag.compiled, "NEON must be compiled");
            kani::assert(flag.runtime, "NEON must be available");
        }
        EligibleBackend::PortableSimd => {
            kani::assert(flag.compiled, "portable SIMD must be compiled");
            kani::assert(flag.runtime, "portable SIMD must be available");
        }
    }
}

fn bounded_usize(max_inclusive: usize) -> usize {
    let value = kani::any::<usize>();
    kani::assume(value <= max_inclusive);
    value
}

fn symbolic_lane_width() -> usize {
    let lane_choice = kani::any::<u8>();
    kani::assume(lane_choice < 3);
    match lane_choice {
        0 => 4,
        1 => 8,
        _ => 16,
    }
}

/// Describes the geometry of a zero-padded SIMD block.
struct PaddedBlock {
    point_count: usize,
    padded_count: usize,
    lanes: usize,
}

fn verify_active_lane_bounds(offset: usize, lane_index: usize, block: &PaddedBlock) {
    kani::assert(
        lane_index < block.lanes,
        "logical lane index must fit the backend lane",
    );
    kani::assert(
        offset + lane_index < block.point_count,
        "logical output write must stay inside the output buffer",
    );
    kani::assert(
        offset + lane_index < block.padded_count,
        "logical output write must correspond to a padded lane",
    );
}

fn verify_active_batch(offset: usize, block: &PaddedBlock) {
    kani::assert(
        offset + block.lanes <= block.padded_count,
        "full SIMD lane load must stay inside the padded block",
    );

    let remaining = lane_output_count(block.point_count, offset, block.lanes);
    kani::assert(
        remaining <= block.lanes,
        "logical output count must fit the lane",
    );
    for lane_index in 0..super::MAX_SIMD_LANES {
        if lane_index < remaining {
            verify_active_lane_bounds(offset, lane_index, block);
        }
    }
}

fn verify_lane_loads(point_count: usize, padded_count: usize, lanes: usize) {
    let block = PaddedBlock {
        point_count,
        padded_count,
        lanes,
    };
    for batch_index in 0..MAX_PROOF_BATCHES {
        let offset = batch_index * block.lanes;
        if offset < block.padded_count {
            verify_active_batch(offset, &block);
        } else {
            kani::assert(
                lane_output_count(block.point_count, offset, block.lanes) == 0,
                "batches beyond the padded block must not write outputs",
            );
        }
    }
}
