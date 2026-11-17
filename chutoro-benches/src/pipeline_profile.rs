//! Opt-in stage profile for the whole CPU clustering pipeline benchmark.
//!
//! The `pipeline-profile` feature installs a local metrics recorder around one
//! untimed `Chutoro::run` and turns the bounded
//! `chutoro.cpu_pipeline.stage_seconds` histogram into CSV rows. The file is
//! written below the conventional benchmark report directory so the stage
//! shares, including the `core_distances` share, sit beside the other
//! benchmark diagnostics.
//!
//! CSV formatting is always compiled and unit-tested; only the recorder-backed
//! measurement depends on the feature. Criterion itself never observes these
//! runs, so profiling cannot bias the recorded benchmark distribution.

use camino::Utf8PathBuf;
use mockable::{DefaultEnv, Env};

use crate::{error::BenchSetupError, report_file::write_report_file};

/// Filename of the pipeline stage profile below the benchmark report directory.
pub const PIPELINE_PROFILE_REPORT: &str = "pipeline_profile.csv";

/// Environment variable overriding the complete profile file path.
pub const PIPELINE_PROFILE_PATH_ENV: &str = "CHUTORO_BENCH_PIPELINE_PROFILE_PATH";

/// Histogram whose bounded `stage` label the profile reads.
#[cfg(feature = "pipeline-profile")]
const STAGE_SECONDS_METRIC: &str = "chutoro.cpu_pipeline.stage_seconds";

/// Label used for the row describing the whole profiled run.
const WHOLE_RUN_STAGE: &str = "whole_run";

/// Header row of the pipeline stage profile.
const PROFILE_HEADER: &str = "point_count,stage,seconds,share\n";

/// One stage's wall-clock time from a profiled run.
#[derive(Clone, Debug, PartialEq)]
pub struct PipelineStageSample {
    /// Bounded pipeline stage label.
    pub stage: String,
    /// Wall-clock seconds the stage took.
    pub seconds: f64,
}

/// Measurements from one profiled pipeline run.
#[derive(Clone, Debug, PartialEq)]
pub struct PipelineProfileSample {
    /// Source length of the profiled run.
    pub point_count: usize,
    /// Wall-clock seconds for the whole pipeline run.
    pub total_seconds: f64,
    /// Per-stage timings recorded during the run.
    pub stages: Vec<PipelineStageSample>,
}

/// Formats the profile samples as CSV with a whole-run row per sample.
///
/// Each stage row carries its share of the whole run, so the `core_distances`
/// share is readable without recomputing it. A zero-length total yields a zero
/// share rather than a non-finite value.
///
/// # Examples
///
/// ```
/// use chutoro_benches::pipeline_profile::{
///     PipelineProfileSample, PipelineStageSample, format_pipeline_profile,
/// };
///
/// let sample = PipelineProfileSample {
///     point_count: 10,
///     total_seconds: 2.0,
///     stages: vec![PipelineStageSample {
///         stage: "core_distances".to_owned(),
///         seconds: 0.5,
///     }],
/// };
/// let csv = format_pipeline_profile(&[sample]);
/// assert!(csv.starts_with("point_count,stage,seconds,share\n"));
/// assert!(csv.contains("10,core_distances,0.500000,0.2500\n"));
/// ```
#[must_use]
pub fn format_pipeline_profile(samples: &[PipelineProfileSample]) -> String {
    let mut csv = String::from(PROFILE_HEADER);
    for sample in samples {
        push_row(
            &mut csv,
            sample.point_count,
            WHOLE_RUN_STAGE,
            sample.total_seconds,
            1.0,
        );
        for stage in &sample.stages {
            push_row(
                &mut csv,
                sample.point_count,
                &stage.stage,
                stage.seconds,
                share_of_total(stage.seconds, sample.total_seconds),
            );
        }
    }
    csv
}

/// Appends one CSV row with a stage's seconds and share.
fn push_row(csv: &mut String, point_count: usize, stage: &str, seconds: f64, share: f64) {
    csv.push_str(&format!("{point_count},{stage},{seconds:.6},{share:.4}\n"));
}

/// Returns `seconds / total`, or zero when the total is not positive.
fn share_of_total(seconds: f64, total: f64) -> f64 {
    if total > 0.0 { seconds / total } else { 0.0 }
}

/// Writes the profile CSV below the benchmark report directory.
///
/// The default destination follows the shared benchmark report convention
/// (`<CARGO_TARGET_DIR>/benchmarks/pipeline_profile.csv`), so it is ignored by
/// version control and discoverable beside the other reports.
/// [`PIPELINE_PROFILE_PATH_ENV`] overrides the complete file path.
///
/// # Errors
///
/// Returns [`BenchSetupError::PipelineProfileReport`] when the report
/// directory or file cannot be written.
pub fn write_pipeline_profile(
    samples: &[PipelineProfileSample],
) -> Result<Utf8PathBuf, BenchSetupError> {
    let contents = format_pipeline_profile(samples);
    write_report_file(&profile_path_with_env(&DefaultEnv), &contents)
        .map_err(BenchSetupError::PipelineProfileReport)
}

/// Resolves the profile destination through an injected environment reader.
///
/// An explicit override wins; otherwise the shared report convention applies.
fn profile_path_with_env(env: &dyn Env) -> Utf8PathBuf {
    env.string(PIPELINE_PROFILE_PATH_ENV).map_or_else(
        || crate::neighbour_scoring::report_path(PIPELINE_PROFILE_REPORT).path(),
        Utf8PathBuf::from,
    )
}

/// Runs the pipeline once under a local metrics recorder and collects timings.
///
/// The run is deliberately outside Criterion so profiling never contributes to
/// the recorded distribution. Only the bounded stage histogram is read; other
/// metrics recorded during the run are ignored.
///
/// # Errors
///
/// Returns [`BenchSetupError::Pipeline`] when the pipeline run fails.
#[cfg(feature = "pipeline-profile")]
pub fn measure_pipeline_stages<D>(
    point_count: usize,
    source: &D,
    chutoro: &chutoro_core::Chutoro,
) -> Result<PipelineProfileSample, BenchSetupError>
where
    D: chutoro_core::DataSource + Sync,
{
    let recorder = metrics_util::debugging::DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let started = std::time::Instant::now();
    let result = metrics::with_local_recorder(&recorder, || chutoro.run(source));
    let total_seconds = started.elapsed().as_secs_f64();
    result?;

    Ok(PipelineProfileSample {
        point_count,
        total_seconds,
        stages: collect_stage_samples(snapshotter.snapshot()),
    })
}

/// Extracts sorted stage timings from a metrics snapshot.
#[cfg(feature = "pipeline-profile")]
fn collect_stage_samples(snapshot: metrics_util::debugging::Snapshot) -> Vec<PipelineStageSample> {
    use metrics_util::debugging::DebugValue;

    let mut stages: Vec<PipelineStageSample> = snapshot
        .into_hashmap()
        .into_iter()
        .filter(|(key, _)| key.key().name() == STAGE_SECONDS_METRIC)
        .filter_map(|(key, (_, _, value))| match value {
            DebugValue::Histogram(samples) => {
                let stage = key
                    .key()
                    .labels()
                    .map(|label| label.value().to_owned())
                    .collect::<Vec<_>>()
                    .join(",");
                let seconds = samples
                    .iter()
                    .map(|sample| sample.into_inner())
                    .sum::<f64>();
                Some(PipelineStageSample { stage, seconds })
            }
            _ => None,
        })
        .collect();
    stages.sort_by(|left, right| left.stage.cmp(&right.stage));
    stages
}
#[cfg(test)]
mod tests {
    //! Tests for pipeline stage profile formatting and measurement.

    use camino::Utf8PathBuf;
    use mockable::MockEnv;
    use rstest::rstest;

    use super::{
        PIPELINE_PROFILE_PATH_ENV, PIPELINE_PROFILE_REPORT, PROFILE_HEADER, PipelineProfileSample,
        PipelineStageSample, format_pipeline_profile, profile_path_with_env,
    };

    /// Builds a sample with a whole-run duration and one core-distance stage.
    fn core_distance_sample() -> PipelineProfileSample {
        PipelineProfileSample {
            point_count: 20,
            total_seconds: 4.0,
            stages: vec![PipelineStageSample {
                stage: "core_distances".to_owned(),
                seconds: 1.0,
            }],
        }
    }

    #[test]
    fn profile_has_one_whole_run_row_and_one_row_per_stage() {
        let csv = format_pipeline_profile(&[core_distance_sample()]);

        assert!(csv.starts_with(PROFILE_HEADER));
        assert!(csv.contains("20,whole_run,4.000000,1.0000\n"));
        assert!(csv.contains("20,core_distances,1.000000,0.2500\n"));
    }

    #[test]
    fn empty_samples_produce_only_the_header() {
        assert_eq!(format_pipeline_profile(&[]), PROFILE_HEADER);
    }

    #[rstest]
    #[case::positive_total(1.0, 4.0, 0.25)]
    #[case::zero_total(1.0, 0.0, 0.0)]
    fn stage_share_is_relative_to_the_whole_run(
        #[case] seconds: f64,
        #[case] total: f64,
        #[case] expected: f64,
    ) {
        let sample = PipelineProfileSample {
            point_count: 1,
            total_seconds: total,
            stages: vec![PipelineStageSample {
                stage: "core_distances".to_owned(),
                seconds,
            }],
        };
        let csv = format_pipeline_profile(&[sample]);

        assert!(csv.contains(&format!("1,core_distances,{seconds:.6},{expected:.4}\n")));
    }

    #[test]
    fn explicit_path_override_wins_over_the_report_convention() {
        let mut env = MockEnv::new();
        env.expect_string().returning(|key| {
            assert_eq!(key, PIPELINE_PROFILE_PATH_ENV);
            Some("/tmp/custom/pipeline_profile.csv".to_owned())
        });

        assert_eq!(
            profile_path_with_env(&env),
            Utf8PathBuf::from("/tmp/custom/pipeline_profile.csv")
        );
    }

    #[test]
    fn missing_override_falls_back_to_the_report_convention() {
        let mut env = MockEnv::new();
        env.expect_string().returning(|key| {
            assert_eq!(key, PIPELINE_PROFILE_PATH_ENV);
            None
        });

        assert_eq!(
            profile_path_with_env(&env).file_name(),
            Some(PIPELINE_PROFILE_REPORT)
        );
    }

    #[cfg(feature = "pipeline-profile")]
    #[test]
    fn profiled_run_records_every_pipeline_stage() {
        use chutoro_core::{ChutoroBuilder, DataSource, ExecutionStrategy, HnswParams};

        use super::measure_pipeline_stages;
        use crate::source::{SyntheticConfig, SyntheticSource};

        let source = SyntheticSource::generate(&SyntheticConfig {
            point_count: 12,
            dimensions: 4,
            seed: 7,
        })
        .expect("synthetic source must generate");
        let params = HnswParams::new(4, 8).expect("parameters must be valid");
        let chutoro = ChutoroBuilder::new()
            .with_min_cluster_size(2)
            .with_execution_strategy(ExecutionStrategy::CpuOnly)
            .with_hnsw_params(params)
            .build()
            .expect("builder must validate");

        let sample = measure_pipeline_stages(source.len(), &source, &chutoro)
            .expect("profiled run must succeed");

        let stages: Vec<&str> = sample
            .stages
            .iter()
            .map(|stage| stage.stage.as_str())
            .collect();
        assert_eq!(
            stages,
            ["core_distances", "kruskal", "mutual_reachability"],
            "the profile must report every bounded pipeline stage"
        );
        assert!(sample.total_seconds > 0.0);
    }
}
