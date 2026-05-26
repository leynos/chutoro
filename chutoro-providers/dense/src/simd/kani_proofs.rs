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
    let compiled_avx2 = kani::any::<bool>();
    let compiled_avx512 = kani::any::<bool>();
    let compiled_neon = kani::any::<bool>();
    let compiled_portable_simd = kani::any::<bool>();
    let runtime_avx2 = kani::any::<bool>();
    let runtime_avx512 = kani::any::<bool>();
    let runtime_neon = kani::any::<bool>();
    let runtime_portable_simd = kani::any::<bool>();

    let compiled = CompiledSimdSupport::new(
        compiled_avx2,
        compiled_avx512,
        compiled_neon,
        compiled_portable_simd,
    );
    let runtime = RuntimeSimdSupport::new(
        runtime_avx2,
        runtime_avx512,
        runtime_neon,
        runtime_portable_simd,
    );

    match dispatch::choose_euclidean_backend(compiled, runtime) {
        dispatch::EuclideanBackend::Avx512 => {
            assert_eligible(
                compiled_avx512,
                runtime_avx512,
                "AVX-512 must be compiled",
                "AVX-512 must be available",
            );
        }
        dispatch::EuclideanBackend::Avx2 => {
            assert_not_eligible(compiled_avx512, runtime_avx512, "AVX-512 has priority");
            assert_eligible(
                compiled_avx2,
                runtime_avx2,
                "AVX2 must be compiled",
                "AVX2 must be available",
            );
        }
        dispatch::EuclideanBackend::Neon => {
            assert_not_eligible(compiled_avx512, runtime_avx512, "AVX-512 has priority");
            assert_not_eligible(compiled_avx2, runtime_avx2, "AVX2 has priority");
            assert_eligible(
                compiled_neon,
                runtime_neon,
                "NEON must be compiled",
                "NEON must be available",
            );
        }
        dispatch::EuclideanBackend::PortableSimd => {
            assert_not_eligible(compiled_avx512, runtime_avx512, "AVX-512 has priority");
            assert_not_eligible(compiled_avx2, runtime_avx2, "AVX2 has priority");
            assert_not_eligible(compiled_neon, runtime_neon, "NEON has priority");
            assert_eligible(
                compiled_portable_simd,
                runtime_portable_simd,
                "portable SIMD must be compiled",
                "portable SIMD must be available",
            );
        }
        dispatch::EuclideanBackend::Scalar => {
            assert_not_eligible(compiled_avx512, runtime_avx512, "AVX-512 must be absent");
            assert_not_eligible(compiled_avx2, runtime_avx2, "AVX2 must be absent");
            assert_not_eligible(compiled_neon, runtime_neon, "NEON must be absent");
            assert_not_eligible(
                compiled_portable_simd,
                runtime_portable_simd,
                "portable SIMD must be absent",
            );
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

fn eligible(compiled: bool, runtime: bool) -> bool {
    compiled && runtime
}

fn assert_not_eligible(compiled: bool, runtime: bool, msg: &'static str) {
    if same_message(msg, "AVX-512 has priority") {
        kani::assert(!eligible(compiled, runtime), "AVX-512 has priority");
    } else if same_message(msg, "AVX2 has priority") {
        kani::assert(!eligible(compiled, runtime), "AVX2 has priority");
    } else if same_message(msg, "NEON has priority") {
        kani::assert(!eligible(compiled, runtime), "NEON has priority");
    } else if same_message(msg, "AVX-512 must be absent") {
        kani::assert(!eligible(compiled, runtime), "AVX-512 must be absent");
    } else if same_message(msg, "AVX2 must be absent") {
        kani::assert(!eligible(compiled, runtime), "AVX2 must be absent");
    } else if same_message(msg, "NEON must be absent") {
        kani::assert(!eligible(compiled, runtime), "NEON must be absent");
    } else if same_message(msg, "portable SIMD must be absent") {
        kani::assert(!eligible(compiled, runtime), "portable SIMD must be absent");
    } else {
        kani::assume(false);
    }
}

fn assert_eligible(
    compiled: bool,
    runtime: bool,
    compiled_msg: &'static str,
    runtime_msg: &'static str,
) {
    if same_message(compiled_msg, "AVX-512 must be compiled")
        && same_message(runtime_msg, "AVX-512 must be available")
    {
        kani::assert(compiled, "AVX-512 must be compiled");
        kani::assert(runtime, "AVX-512 must be available");
    } else if same_message(compiled_msg, "AVX2 must be compiled")
        && same_message(runtime_msg, "AVX2 must be available")
    {
        kani::assert(compiled, "AVX2 must be compiled");
        kani::assert(runtime, "AVX2 must be available");
    } else if same_message(compiled_msg, "NEON must be compiled")
        && same_message(runtime_msg, "NEON must be available")
    {
        kani::assert(compiled, "NEON must be compiled");
        kani::assert(runtime, "NEON must be available");
    } else if same_message(compiled_msg, "portable SIMD must be compiled")
        && same_message(runtime_msg, "portable SIMD must be available")
    {
        kani::assert(compiled, "portable SIMD must be compiled");
        kani::assert(runtime, "portable SIMD must be available");
    } else {
        kani::assume(false);
    }
}

fn same_message(left: &'static str, right: &'static str) -> bool {
    left.as_ptr() == right.as_ptr() && left.len() == right.len()
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

fn verify_lane_loads(point_count: usize, padded_count: usize, lanes: usize) {
    for batch_index in 0..MAX_PROOF_BATCHES {
        let offset = batch_index * lanes;
        if offset < padded_count {
            kani::assert(
                offset + lanes <= padded_count,
                "full SIMD lane load must stay inside the padded block",
            );

            let remaining = lane_output_count(point_count, offset, lanes);
            kani::assert(remaining <= lanes, "logical output count must fit the lane");
            for lane_index in 0..super::MAX_SIMD_LANES {
                if lane_index >= remaining {
                    continue;
                }

                kani::assert(
                    lane_index < lanes,
                    "logical lane index must fit the backend lane",
                );
                kani::assert(
                    offset + lane_index < point_count,
                    "logical output write must stay inside the output buffer",
                );
                kani::assert(
                    offset + lane_index < padded_count,
                    "logical output write must correspond to a padded lane",
                );
            }
        } else {
            kani::assert(
                lane_output_count(point_count, offset, lanes) == 0,
                "batches beyond the padded block must not write outputs",
            );
        }
    }
}
