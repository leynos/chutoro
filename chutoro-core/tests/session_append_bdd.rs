//! Behavioural tests for append-oriented clustering sessions.

use std::{num::ParseIntError, sync::Arc};

use rstest::fixture;
use rstest_bdd::StepResult;
use rstest_bdd_macros::{given, scenario, then, when};

use chutoro_core::{
    ChutoroBuilder, ChutoroError, ClusteringSession, DataSource, DataSourceError, MetricDescriptor,
};

#[derive(Debug)]
enum BddStepError {
    Parse(ParseIntError),
}

impl std::fmt::Display for BddStepError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(error) => write!(formatter, "invalid BDD index list: {error}"),
        }
    }
}

impl std::error::Error for BddStepError {}

impl From<ParseIntError> for BddStepError {
    fn from(error: ParseIntError) -> Self {
        Self::Parse(error)
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

    fn name(&self) -> &str {
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
        Ok((left - right).abs())
    }

    fn metric_descriptor(&self) -> MetricDescriptor {
        MetricDescriptor::new("session-append-bdd:abs")
    }
}

#[fixture]
fn world() -> SessionAppendWorld {
    let source = Arc::new(SessionAppendSource {
        values: vec![0.0, 1.0, 4.0],
    });
    let session = ChutoroBuilder::new()
        .build_session(source)
        .expect("behavioural session fixture must build");

    SessionAppendWorld {
        session,
        last_error: None,
    }
}

#[given("an empty clustering session")]
fn empty_session(world: &mut SessionAppendWorld) {
    assert_eq!(world.session.point_count(), 0);
    assert_eq!(world.session.snapshot_version(), 0);
    assert!(world.last_error.is_none());
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

#[scenario(
    path = "tests/features/session_append.feature",
    name = "Appending valid source indices"
)]
fn append_valid_source_indices(_world: SessionAppendWorld) {}

#[scenario(
    path = "tests/features/session_append.feature",
    name = "Duplicate index rejection"
)]
fn duplicate_index_rejection(_world: SessionAppendWorld) {}

#[scenario(
    path = "tests/features/session_append.feature",
    name = "Out-of-bounds index rejection"
)]
fn out_of_bounds_index_rejection(_world: SessionAppendWorld) {}

#[scenario(
    path = "tests/features/session_append.feature",
    name = "Empty index list no-op"
)]
fn empty_index_list_no_op(_world: SessionAppendWorld) {}

#[scenario(
    path = "tests/features/session_append.feature",
    name = "Snapshot version immutability across multiple appends"
)]
fn snapshot_version_immutability_across_multiple_appends(_world: SessionAppendWorld) {}
