//! Behavioural acceptance tests for [`ClusteringSession::append`], defined in
//! `chutoro-core/src/session/mod.rs`.
//!
//! These tests use the [`rstest-bdd`] framework driven by the Gherkin feature
//! file at `chutoro-core/tests/features/session_append.feature`.  They sit
//! above the unit-test layer in the testing hierarchy: each scenario exercises
//! the public `append` API end-to-end through `ChutoroBuilder`, asserting
//! observable session state (`point_count`, `snapshot_version`, error codes)
//! without access to internal fields.

use std::{num::ParseIntError, sync::Arc};

use rstest::fixture;
use rstest_bdd::StepResult;
use rstest_bdd_macros::{given, scenario, then, when};

use chutoro_core::{
    ChutoroBuilder, ChutoroError, ClusteringSession, DataSource, DataSourceError, MetricDescriptor,
};

include!(concat!(env!("OUT_DIR"), "/users_guide_session_refresh.rs"));

#[derive(Debug)]
enum BddStepError {
    Parse(ParseIntError),
    Session(ChutoroError),
}

impl std::fmt::Display for BddStepError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(error) => write!(formatter, "invalid BDD index list: {error}"),
            Self::Session(error) => write!(formatter, "failed to create BDD session: {error}"),
        }
    }
}

impl std::error::Error for BddStepError {}

impl From<ParseIntError> for BddStepError {
    fn from(error: ParseIntError) -> Self {
        Self::Parse(error)
    }
}

impl From<ChutoroError> for BddStepError {
    fn from(error: ChutoroError) -> Self {
        Self::Session(error)
    }
}

#[derive(Clone, Debug)]
struct SessionAppendSource {
    values: Vec<f32>,
}

#[derive(Debug)]
struct SessionAppendWorld {
    session: ClusteringSession<SessionAppendSource>,
    last_error: Option<ChutoroError>,
}

impl DataSource for SessionAppendSource {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn name(&self) -> &'static str {
        "session-append-bdd"
    }

    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let left = self
            .values
            .get(i)
            .ok_or(DataSourceError::OutOfBounds { index: i })?;
        let right = self
            .values
            .get(j)
            .ok_or(DataSourceError::OutOfBounds { index: j })?;
        Ok(left.mul_add(1.0, std::ops::Neg::neg(*right)).abs())
    }

    fn metric_descriptor(&self) -> MetricDescriptor {
        MetricDescriptor::new("session-append-bdd:abs")
    }
}

/// Builds a session over `len` zero-based scalar points.
fn build_world(
    len: usize,
    min_cluster_size: usize,
) -> StepResult<SessionAppendWorld, BddStepError> {
    let values = (0..len)
        .map(|value| u16::try_from(value).map_or(f32::MAX, f32::from))
        .collect();
    let source = Arc::new(SessionAppendSource { values });
    let session = ChutoroBuilder::new()
        .with_min_cluster_size(min_cluster_size)
        .build_session(source)?;

    Ok(SessionAppendWorld {
        session,
        last_error: None,
    })
}

#[fixture]
fn world() -> StepResult<SessionAppendWorld, BddStepError> {
    build_world(3, 1)
}

#[given("an empty clustering session")]
fn empty_session(world: &mut SessionAppendWorld) {
    assert_eq!(world.session.point_count(), 0);
    assert_eq!(world.session.snapshot_version(), 0);
    assert!(world.last_error.is_none());
}

#[given("a clustering session over {count:usize} points with min cluster size {min:usize}")]
fn sized_session(
    world: &mut SessionAppendWorld,
    count: usize,
    min: usize,
) -> StepResult<(), BddStepError> {
    *world = build_world(count, min)?;
    Ok(())
}

#[when("I append source indices {indices:string}")]
fn append_source_indices(
    world: &mut SessionAppendWorld,
    indices: &str,
) -> StepResult<(), BddStepError> {
    let parsed = if indices.trim().is_empty() {
        Vec::new()
    } else {
        indices
            .split(',')
            .map(str::trim)
            .map(str::parse::<usize>)
            .collect::<Result<Vec<_>, _>>()?
    };

    world.last_error = world.session.append(&parsed).err();
    Ok(())
}

#[when("I recompute core distances")]
fn recompute_core_distances(world: &mut SessionAppendWorld) {
    world.last_error = world.session.recompute_core_distances().err();
}

#[when("I refresh the session")]
fn refresh_session(world: &mut SessionAppendWorld) {
    world.last_error = world.session.refresh().err();
}

#[then("the label snapshot has length {count:usize}")]
fn label_snapshot_has_length(world: &SessionAppendWorld, count: usize) {
    assert!(world.last_error.is_none(), "refresh must have succeeded");
    assert_eq!(world.session.labels().len(), count);
}

#[then("the label snapshot is empty")]
fn label_snapshot_is_empty(world: &SessionAppendWorld) {
    assert!(world.last_error.is_none(), "refresh must have succeeded");
    assert_eq!(*world.session.labels(), Vec::<usize>::new());
}

#[then("the append succeeds")]
fn append_succeeds(world: &SessionAppendWorld) {
    assert!(world.last_error.is_none());
}

#[then("the append is rejected")]
fn append_is_rejected(world: &SessionAppendWorld) {
    assert!(world.last_error.is_some());
}

#[then("the session contains {count:usize} points")]
fn session_contains_points(world: &SessionAppendWorld, count: usize) {
    assert_eq!(world.session.point_count(), count);
}

#[then("the snapshot version is {version:u64}")]
fn snapshot_version_is(world: &SessionAppendWorld, version: u64) {
    assert_eq!(world.session.snapshot_version(), version);
}

#[when("I run the published users' guide refresh example")]
fn run_published_users_guide_example(world: &mut SessionAppendWorld) {
    world.last_error = published_example::run().err();
}

#[then("the published example succeeds")]
fn published_example_succeeds(world: &SessionAppendWorld) {
    assert!(
        world.last_error.is_none(),
        "the published users' guide example must succeed, got {:?}",
        world.last_error
    );
}

#[then("source index {index:usize} has core distance {distance:f32}")]
fn source_index_has_core_distance(world: &SessionAppendWorld, index: usize, distance: f32) {
    assert_eq!(world.session.core_distance(index), Some(distance));
}

#[scenario(
    path = "tests/features/session_append.feature",
    name = "Appending valid source indices"
)]
fn append_valid_source_indices(
    _world: StepResult<SessionAppendWorld, BddStepError>,
) -> StepResult<(), BddStepError> {
    Ok(())
}

#[scenario(
    path = "tests/features/session_append.feature",
    name = "Duplicate index rejection"
)]
fn duplicate_index_rejection(
    _world: StepResult<SessionAppendWorld, BddStepError>,
) -> StepResult<(), BddStepError> {
    Ok(())
}

#[scenario(
    path = "tests/features/session_append.feature",
    name = "Out-of-bounds index rejection"
)]
fn out_of_bounds_index_rejection(
    _world: StepResult<SessionAppendWorld, BddStepError>,
) -> StepResult<(), BddStepError> {
    Ok(())
}

#[scenario(
    path = "tests/features/session_append.feature",
    name = "Empty index list no-op"
)]
fn empty_index_list_no_op(
    _world: StepResult<SessionAppendWorld, BddStepError>,
) -> StepResult<(), BddStepError> {
    Ok(())
}

#[scenario(
    path = "tests/features/session_append.feature",
    name = "Snapshot version immutability across multiple appends"
)]
fn snapshot_version_immutability_across_multiple_appends(
    _world: StepResult<SessionAppendWorld, BddStepError>,
) -> StepResult<(), BddStepError> {
    Ok(())
}

#[scenario(
    path = "tests/features/session_append.feature",
    name = "Recomputing core distances after append"
)]
fn recomputing_core_distances_after_append(
    _world: StepResult<SessionAppendWorld, BddStepError>,
) -> StepResult<(), BddStepError> {
    Ok(())
}

#[scenario(
    path = "tests/features/session_append.feature",
    name = "The published users' guide example completes"
)]
fn the_published_users_guide_example_completes(
    _world: StepResult<SessionAppendWorld, BddStepError>,
) -> StepResult<(), BddStepError> {
    Ok(())
}

#[scenario(
    path = "tests/features/session_append.feature",
    name = "Refreshing a populated session publishes labels for every point"
)]
fn refreshing_a_populated_session_publishes_labels(
    _world: StepResult<SessionAppendWorld, BddStepError>,
) -> StepResult<(), BddStepError> {
    Ok(())
}

#[scenario(
    path = "tests/features/session_append.feature",
    name = "Refreshing an empty session bumps the version, empty labels"
)]
fn refreshing_an_empty_session_bumps_the_version(
    _world: StepResult<SessionAppendWorld, BddStepError>,
) -> StepResult<(), BddStepError> {
    Ok(())
}
