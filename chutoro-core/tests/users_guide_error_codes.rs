//! Pin the subsystem error-code reference in the user guide.
//!
//! The guide documents a machine-readable surface, so the test loads the
//! shipped text and checks that it neither omits a subsystem code nor names a
//! code that no public code enum can produce.

#![cfg(feature = "cpu")]

use chutoro_core::{
    ChutoroErrorCode, DataSourceErrorCode, HierarchyErrorCode, HnswErrorCode, MstErrorCode,
};

const USERS_GUIDE: &str = include_str!("../../docs/users-guide.md");

/// Heading that opens the code reference under test.
const SECTION_HEADING: &str = "### Subsystem error codes";

/// Every `MstErrorCode` variant, mirroring the integration error tests.
const MST_CODES: [MstErrorCode; 5] = [
    MstErrorCode::EmptyGraph,
    MstErrorCode::InvalidNodeId,
    MstErrorCode::NonFiniteWeight,
    MstErrorCode::LockPoisoned,
    MstErrorCode::InvariantViolation,
];

/// Every `HnswErrorCode` variant.
const HNSW_CODES: [HnswErrorCode; 8] = [
    HnswErrorCode::EmptyBuild,
    HnswErrorCode::InvalidParameters,
    HnswErrorCode::DuplicateNode,
    HnswErrorCode::GraphEmpty,
    HnswErrorCode::GraphInvariantViolation,
    HnswErrorCode::NonFiniteDistance,
    HnswErrorCode::LockPoisoned,
    HnswErrorCode::DataSource,
];

/// Every `HierarchyErrorCode` variant.
const HIERARCHY_CODES: [HierarchyErrorCode; 7] = [
    HierarchyErrorCode::EmptyDataset,
    HierarchyErrorCode::MinClusterSizeTooLarge,
    HierarchyErrorCode::InvalidEdgeWeight,
    HierarchyErrorCode::InvalidEdgeEndpoint,
    HierarchyErrorCode::InvalidForestReference,
    HierarchyErrorCode::InvalidClusterReference,
    HierarchyErrorCode::InvalidPointReference,
];

/// Every `ChutoroErrorCode` variant.
const CHUTORO_CODES: [ChutoroErrorCode; 9] = [
    ChutoroErrorCode::InvalidMinClusterSize,
    ChutoroErrorCode::EmptySource,
    ChutoroErrorCode::InsufficientItems,
    ChutoroErrorCode::BackendUnavailable,
    ChutoroErrorCode::DataSourceFailure,
    ChutoroErrorCode::CpuHnswFailure,
    ChutoroErrorCode::CpuMstFailure,
    ChutoroErrorCode::CpuHierarchyFailure,
    ChutoroErrorCode::MemoryLimitExceeded,
];

/// Every `DataSourceErrorCode` variant.
const DATA_SOURCE_CODES: [DataSourceErrorCode; 5] = [
    DataSourceErrorCode::OutOfBounds,
    DataSourceErrorCode::OutputLengthMismatch,
    DataSourceErrorCode::DimensionMismatch,
    DataSourceErrorCode::EmptyData,
    DataSourceErrorCode::ZeroDimension,
];

/// Returns the Markdown heading level when a line is an ATX heading.
fn heading_level(line: &str) -> Option<usize> {
    let level = line.bytes().take_while(|byte| *byte == b'#').count();
    let has_space = line.as_bytes().get(level) == Some(&b' ');
    ((1..=6).contains(&level) && has_space).then_some(level)
}

/// Returns the reference section, stopping at the next heading of the same or
/// a higher level.
///
/// The section owns `####` child headings, so only a `###` or `##` heading can
/// end it.
fn subsystem_section() -> Option<&'static str> {
    let (_, rest) = USERS_GUIDE.split_once(SECTION_HEADING)?;
    let end = rest
        .match_indices('\n')
        .map(|(index, _)| index + 1)
        .chain(std::iter::once(0))
        .filter(|start| {
            rest.get(*start..)
                .and_then(|tail| tail.lines().next())
                .and_then(heading_level)
                .is_some_and(|level| level <= 3)
        })
        .min()
        .unwrap_or(rest.len());
    Some(rest.get(..end).unwrap_or(rest).trim_end())
}

/// Reports whether a backticked token looks like a stable error code.
fn is_upper_snake_case(token: &str) -> bool {
    let mut has_underscore = false;
    let mut has_letter = false;
    for byte in token.bytes() {
        match byte {
            b'A'..=b'Z' => has_letter = true,
            b'0'..=b'9' => {}
            b'_' => has_underscore = true,
            _ => return false,
        }
    }
    has_letter && has_underscore
}

/// Returns the upper-snake-case tokens wrapped in single backticks.
fn backticked_upper_snake_tokens(section: &'static str) -> Vec<&'static str> {
    section
        .split('`')
        .skip(1)
        .step_by(2)
        .filter(|token| is_upper_snake_case(token))
        .collect()
}

/// Collects every stable string a public error-code enum can produce.
fn known_code_strings() -> Vec<&'static str> {
    MST_CODES
        .iter()
        .map(|code| (*code).as_str())
        .chain(HNSW_CODES.iter().map(|code| (*code).as_str()))
        .chain(HIERARCHY_CODES.iter().map(|code| (*code).as_str()))
        .chain(CHUTORO_CODES.iter().map(|code| (*code).as_str()))
        .chain(DATA_SOURCE_CODES.iter().map(|code| (*code).as_str()))
        .collect()
}

/// Asserts that each code is documented as a backticked token.
fn assert_codes_documented(codes: &[&str]) -> Result<(), String> {
    let section =
        subsystem_section().ok_or_else(|| format!("guide must contain {SECTION_HEADING}"))?;
    for code in codes {
        let token = format!("`{code}`");
        if !section.contains(&token) {
            return Err(format!("guide must document {token}"));
        }
    }
    Ok(())
}

/// Verifies the guide lists every MST error code.
#[test]
fn documents_every_mst_error_code() {
    let codes: Vec<&str> = MST_CODES.iter().map(|code| (*code).as_str()).collect();
    assert_codes_documented(&codes).expect("every MST code must appear in the guide");
}

/// Verifies the guide lists every HNSW error code.
#[test]
fn documents_every_hnsw_error_code() {
    let codes: Vec<&str> = HNSW_CODES.iter().map(|code| (*code).as_str()).collect();
    assert_codes_documented(&codes).expect("every HNSW code must appear in the guide");
}

/// Verifies the guide lists every hierarchy error code.
#[test]
fn documents_every_hierarchy_error_code() {
    let codes: Vec<&str> = HIERARCHY_CODES
        .iter()
        .map(|code| (*code).as_str())
        .collect();
    assert_codes_documented(&codes).expect("every hierarchy code must appear in the guide");
}

/// Verifies the section does not name a code that no enum can produce.
#[test]
fn names_only_known_error_codes() {
    let section = subsystem_section().expect("guide must contain the subsystem error-code section");
    let known = known_code_strings();
    for token in backticked_upper_snake_tokens(section) {
        assert!(
            known.contains(&token),
            "the guide names `{token}`, which no public error-code enum produces"
        );
    }
}
