//! Support code for the neighbour-scoring Criterion harness.
//!
//! The module keeps fixture construction and diagnostic reporting separate from
//! the benchmark body so the measured path stays easy to inspect.

use std::{io, num::TryFromIntError, sync::Arc, time::Instant};

use super::{
    BUILD_PROFILE_REPORT, BuildProfileReportRow, LANE_REPORT, LaneUtilisationReportRow,
    REPORT_DIR_NAME, ReportTarget, report_path_value, sorted_median,
    write_build_profile_report_csv, write_lane_utilisation_report_csv,
};
use crate::source::{SyntheticConfig, SyntheticError, SyntheticSource};
use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array};
use arrow_schema::{DataType, Field};
use camino::{Utf8Path, Utf8PathBuf};
use cap_std::{ambient_authority, fs_utf8::Dir};
use chutoro_core::{CpuHnsw, DataSourceError, HnswError, HnswParams};
use chutoro_providers_dense::{DenseMatrixProvider, DenseMatrixProviderError};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use thiserror::Error;

use super::{ProfilingError, ProfilingSource, all_buckets};

const BENCH_ROW_COUNT: usize = 1_025;
const BENCH_SEED: u64 = 0xC4A7_0203_0000_0231;
pub(super) const DEFAULT_BUILD_PROFILE_POINT_COUNTS: &[usize] = &[10_000, 100_000];
pub(super) const DEFAULT_BUILD_PROFILE_DIMENSION: usize = 128;

#[derive(Debug, Error)]
pub(super) enum BenchError {
    #[error("data source error: {0}")]
    DataSource(#[from] DataSourceError),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("build profile statistics failed: {0}")]
    BuildProfileStats(#[from] ProfilingError),
    #[error("dimension {dimension} does not fit {target}: {source}")]
    DimensionConversion {
        dimension: usize,
        target: &'static str,
        source: TryFromIntError,
    },
    #[error("candidate count {candidate_count} does not fit u64: {source}")]
    CandidateCountConversion {
        candidate_count: usize,
        source: TryFromIntError,
    },
    #[error("dense provider setup failed: {0}")]
    DenseProvider(#[from] DenseMatrixProviderError),
    #[error("HNSW parameter setup failed: {source}")]
    HnswParams {
        #[source]
        source: HnswError,
    },
    #[error("synthetic source setup failed: {0}")]
    SyntheticSource(#[from] SyntheticError),
    #[error("HNSW build failed: {source}")]
    HnswBuild {
        #[source]
        source: HnswError,
    },
}

pub(super) type BenchResult<T> = Result<T, BenchError>;

#[derive(Debug)]
pub(super) struct ScoringFixture {
    pub(super) provider: DenseMatrixProvider,
    pub(super) candidates: Vec<usize>,
}

fn open_report_dir(report_parent_dir: &Utf8Path) -> BenchResult<Dir> {
    let target_dir = Dir::open_ambient_dir(report_parent_dir, ambient_authority())?;
    target_dir.create_dir_all(REPORT_DIR_NAME)?;
    Ok(target_dir.open_dir(REPORT_DIR_NAME)?)
}

fn make_values(row_count: usize, dimension: usize) -> BenchResult<Vec<f32>> {
    let seed_dimension =
        u64::try_from(dimension).map_err(|source| BenchError::DimensionConversion {
            dimension,
            target: "u64",
            source,
        })?;
    let mut rng = SmallRng::seed_from_u64(BENCH_SEED ^ seed_dimension);
    let mut values = Vec::with_capacity(row_count.saturating_mul(dimension));
    for _ in 0..row_count {
        for _ in 0..dimension {
            values.push(rng.gen_range(-1.0_f32..1.0_f32));
        }
    }
    Ok(values)
}

fn make_provider(row_count: usize, dimension: usize) -> BenchResult<DenseMatrixProvider> {
    let width = i32::try_from(dimension).map_err(|source| BenchError::DimensionConversion {
        dimension,
        target: "i32",
        source,
    })?;
    let values = Float32Array::from_iter_values(make_values(row_count, dimension)?);
    let array = FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        width,
        Arc::new(values) as ArrayRef,
        None,
    );
    Ok(DenseMatrixProvider::try_from_fixed_size_list(
        "neighbour-scoring",
        &array,
    )?)
}

pub(super) fn make_fixture(
    dimension: usize,
    candidate_count: usize,
) -> BenchResult<ScoringFixture> {
    let required_rows = candidate_count.saturating_add(1);
    let row_count = BENCH_ROW_COUNT.max(required_rows);
    let provider = make_provider(row_count, dimension)?;
    let candidates = (1..=candidate_count).collect();
    Ok(ScoringFixture {
        provider,
        candidates,
    })
}

pub(super) fn write_lane_utilisation_report(
    report_parent_dir: &Utf8Path,
) -> BenchResult<Utf8PathBuf> {
    let report_dir = open_report_dir(report_parent_dir)?;
    let target = report_path_value(report_parent_dir, LANE_REPORT);
    let mut file = report_dir.create(LANE_REPORT)?;
    write_lane_utilisation_report_csv(
        &mut file,
        all_buckets().map(|bucket| LaneUtilisationReportRow {
            bucket_kind: bucket.kind_name(),
            candidate_count: bucket.size(),
        }),
    )?;
    Ok(target.path())
}

fn hnsw_params() -> BenchResult<HnswParams> {
    HnswParams::new(16, 32).map_err(|source| BenchError::HnswParams { source })
}

fn profile_source(
    point_count: usize,
    dimension: usize,
) -> BenchResult<ProfilingSource<SyntheticSource>> {
    let source = SyntheticSource::generate(&SyntheticConfig {
        point_count,
        dimensions: dimension,
        seed: BENCH_SEED,
    })?;
    Ok(ProfilingSource::new(source))
}

pub(super) fn write_build_profile_report(
    report_parent_dir: Option<&Utf8Path>,
) -> BenchResult<Option<ReportTarget>> {
    write_build_profile_report_with(
        report_parent_dir,
        write_default_build_profile_report_for_point_counts,
    )
}

fn write_build_profile_report_with(
    report_parent_dir: Option<&Utf8Path>,
    writer: impl FnOnce(&Utf8Path, &[usize], usize) -> BenchResult<ReportTarget>,
) -> BenchResult<Option<ReportTarget>> {
    report_parent_dir
        .map(|configured_report_parent_dir| {
            writer(
                configured_report_parent_dir,
                DEFAULT_BUILD_PROFILE_POINT_COUNTS,
                DEFAULT_BUILD_PROFILE_DIMENSION,
            )
        })
        .transpose()
}

fn write_default_build_profile_report_for_point_counts(
    report_parent_dir: &Utf8Path,
    point_counts: &[usize],
    dimension: usize,
) -> BenchResult<ReportTarget> {
    let report_target = report_path_value(report_parent_dir, BUILD_PROFILE_REPORT);
    write_build_profile_report_for_point_counts(&report_target, point_counts, dimension)
}

pub(super) fn write_build_profile_report_for_point_counts(
    report_target: &ReportTarget,
    point_counts: &[usize],
    dimension: usize,
) -> BenchResult<ReportTarget> {
    let mut report_rows = Vec::new();

    for &point_count in point_counts {
        let source = profile_source(point_count, dimension)?;
        let params = hnsw_params()?;
        let started = Instant::now();
        CpuHnsw::build(&source, params)
            .map_err(|hnsw_error| BenchError::HnswBuild { source: hnsw_error })?;
        let build_elapsed = started.elapsed();
        let stats = source.take_snapshot()?;
        let mut batch_sizes = stats.batch_sizes;
        batch_sizes.sort_unstable();
        let min_batch = batch_sizes.first().copied().unwrap_or(0);
        let max_batch = batch_sizes.last().copied().unwrap_or(0);
        let median_batch = sorted_median(&batch_sizes);
        report_rows.push(BuildProfileReportRow {
            point_count,
            dimension,
            build_elapsed,
            batch_scoring_time: stats.batch_scoring_time,
            batch_calls: stats.batch_calls,
            scalar_calls: stats.scalar_calls,
            total_batch_candidates: stats.total_batch_candidates,
            min_batch,
            max_batch,
            median_batch,
        });
    }
    let report_dir = open_report_dir(report_target.report_parent_dir())?;
    let mut file = report_dir.create(report_target.filename())?;
    write_build_profile_report_csv(&mut file, report_rows)?;
    Ok(report_target.clone())
}

#[cfg(test)]
#[path = "benchmark_support_tests.rs"]
mod tests;
