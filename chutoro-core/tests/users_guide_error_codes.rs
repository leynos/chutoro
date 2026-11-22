//! Pin the subsystem error-code reference in the user guide.
//!
//! The guide documents a machine-readable surface, so the test loads the
//! shipped text and checks that it neither omits a subsystem code nor names a
//! code that no public code enum can produce.

#![cfg(feature = "cpu")]

use chutoro_core::{
    ChutoroErrorCode, DataSourceErrorCode, HierarchyErrorCode, HnswErrorCode, MstErrorCode,
};
use rstest::rstest;

const USERS_GUIDE: &str = include_str!("../../docs/users-guide.md");

/// Heading that opens the code reference under test.
const SECTION_HEADING: &str = "### Subsystem error codes";

/// Heading that opens the MST code table.
const MST_HEADING: &str = "#### MST errors";

/// Heading that opens the HNSW code table.
const HNSW_HEADING: &str = "#### HNSW errors";

/// Heading that opens the hierarchy code table.
const HIERARCHY_HEADING: &str = "#### Hierarchy errors";

/// Collects the stable strings of every `MstErrorCode` variant.
fn mst_code_strings() -> Vec<&'static str> {
    MstErrorCode::ALL.iter().map(|code| code.as_str()).collect()
}

/// Collects the stable strings of every `HnswErrorCode` variant.
fn hnsw_code_strings() -> Vec<&'static str> {
    HnswErrorCode::ALL
        .iter()
        .map(|code| code.as_str())
        .collect()
}

/// Collects the stable strings of every `HierarchyErrorCode` variant.
fn hierarchy_code_strings() -> Vec<&'static str> {
    HierarchyErrorCode::ALL
        .iter()
        .map(|code| code.as_str())
        .collect()
}

/// Collects the stable strings of every `ChutoroErrorCode` variant.
fn chutoro_code_strings() -> Vec<&'static str> {
    ChutoroErrorCode::ALL
        .iter()
        .map(|code| code.as_str())
        .collect()
}

/// Collects the stable strings of every `DataSourceErrorCode` variant.
fn data_source_code_strings() -> Vec<&'static str> {
    DataSourceErrorCode::ALL
        .iter()
        .map(|code| code.as_str())
        .collect()
}

/// Returns the Markdown heading level when a line is an ATX heading.
fn heading_level(line: &str) -> Option<usize> {
    let level = line.bytes().take_while(|byte| *byte == b'#').count();
    let has_space = line.as_bytes().get(level) == Some(&b' ');
    ((1..=6).contains(&level) && has_space).then_some(level)
}

/// Returns the block opened by `heading`, stopping at the next heading of the
/// same or a higher level.
fn heading_block(heading: &'static str) -> Option<&'static str> {
    let (_, rest) = USERS_GUIDE.split_once(heading)?;
    let level = heading_level(heading)?;
    let end = rest
        .match_indices('\n')
        .map(|(index, _)| index + 1)
        .chain(std::iter::once(0))
        .filter(|start| {
            rest.get(*start..)
                .and_then(|tail| tail.lines().next())
                .and_then(heading_level)
                .is_some_and(|found| found <= level)
        })
        .min()
        .unwrap_or(rest.len());
    Some(rest.get(..end).unwrap_or(rest).trim_end())
}

/// Returns the reference section, stopping at the next heading of the same or
/// a higher level.
///
/// The section owns `####` child headings, so only a `###` or `##` heading can
/// end it.
fn subsystem_section() -> Option<&'static str> {
    heading_block(SECTION_HEADING)
}

/// Returns the table rows inside the block opened by `heading`.
///
/// Prose in the same subsection does not count, so a code must appear in a row
/// of that subsystem's own table rather than elsewhere in the section.
fn table_rows(heading: &'static str) -> Option<Vec<&'static str>> {
    let block = heading_block(heading)?;
    Some(block.lines().filter(|line| line.starts_with('|')).collect())
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
///
/// Reading the macro-generated `ALL` slices keeps this list in step with the
/// enums, so a new variant is covered without editing the test.
fn known_code_strings() -> Vec<&'static str> {
    MstErrorCode::ALL
        .iter()
        .map(|code| code.as_str())
        .chain(HnswErrorCode::ALL.iter().map(|code| code.as_str()))
        .chain(HierarchyErrorCode::ALL.iter().map(|code| code.as_str()))
        .chain(ChutoroErrorCode::ALL.iter().map(|code| code.as_str()))
        .chain(DataSourceErrorCode::ALL.iter().map(|code| code.as_str()))
        .collect()
}

/// Splits a Markdown table row into trimmed cells, dropping the outer pipes.
fn row_cells(row: &'static str) -> Vec<&'static str> {
    row.trim()
        .trim_matches('|')
        .split('|')
        .map(str::trim)
        .collect()
}

/// Returns the code inside a single-code cell such as `` `EMPTY_GRAPH` ``.
fn cell_code(cell: &'static str) -> Option<&'static str> {
    let inner = cell.strip_prefix('`')?.strip_suffix('`')?;
    is_upper_snake_case(inner).then_some(inner)
}

/// Returns the codes listed in the table's `as_str()` column.
///
/// A code counts only when it fills the whole cell, so mentioning another
/// code in a description cannot stand in for that code's own row.
fn tabulated_codes(rows: &[&'static str]) -> Vec<&'static str> {
    let column = rows
        .first()
        .map(|header| row_cells(header))
        .and_then(|cells| cells.iter().position(|cell| *cell == "`as_str()` value"));
    let Some(column_index) = column else {
        return Vec::new();
    };
    rows.iter()
        .filter_map(|row| row_cells(row).get(column_index).copied())
        .filter_map(cell_code)
        .collect()
}

/// Returns the first code that is not the sole content of an `as_str()` cell.
fn missing_code<'a>(rows: &[&'static str], codes: &'a [&str]) -> Option<&'a str> {
    let tabulated = tabulated_codes(rows);
    codes.iter().copied().find(|code| !tabulated.contains(code))
}

/// Returns the first stable string that two variants of one enum share.
fn duplicate_code_string<'a>(codes: &[&'a str]) -> Option<&'a str> {
    codes
        .iter()
        .enumerate()
        .find(|(index, code)| {
            codes
                .get(..*index)
                .is_some_and(|earlier| earlier.contains(code))
        })
        .map(|(_, code)| *code)
}

/// Returns the first table token that belongs to another subsystem.
///
/// The section-wide check accepts any token some enum produces, so this rule
/// is what catches a code copied into the wrong table.
fn foreign_code(rows: &[&'static str], codes: &[&str]) -> Option<&'static str> {
    rows.iter()
        .flat_map(|row| backticked_upper_snake_tokens(row))
        .find(|token| !codes.contains(token))
}

/// Asserts that a table names exactly the codes of one subsystem.
///
/// Every code must be the sole content of an `as_str()` cell, and every
/// upper-snake-case token in a row must be one of that subsystem's codes.
fn assert_codes_tabulated(heading: &'static str, codes: &[&str]) -> Result<(), String> {
    let rows = table_rows(heading).ok_or_else(|| format!("guide must contain {heading}"))?;
    missing_code(&rows, codes)
        .map(|code| format!("{heading} must tabulate `{code}`"))
        .or_else(|| {
            foreign_code(&rows, codes).map(|token| {
                format!("{heading} names `{token}`, which is not a code of that subsystem")
            })
        })
        .map_or(Ok(()), Err)
}

/// Verifies the guide lists every code of each subsystem in its own table.
#[rstest]
#[case::mst(MST_HEADING, mst_code_strings())]
#[case::hnsw(HNSW_HEADING, hnsw_code_strings())]
#[case::hierarchy(HIERARCHY_HEADING, hierarchy_code_strings())]
fn documents_every_subsystem_error_code(
    #[case] heading: &'static str,
    #[case] codes: Vec<&'static str>,
) {
    assert_codes_tabulated(heading, &codes)
        .expect("every subsystem code must appear in its subsystem's table");
}

/// Verifies that each enum maps its variants to distinct stable strings.
///
/// Distinct strings let one table row satisfy exactly one variant, which the
/// per-code row check relies on.
#[rstest]
#[case::mst(mst_code_strings())]
#[case::hnsw(hnsw_code_strings())]
#[case::hierarchy(hierarchy_code_strings())]
#[case::chutoro(chutoro_code_strings())]
#[case::data_source(data_source_code_strings())]
fn code_strings_are_unique_per_enum(#[case] codes: Vec<&'static str>) {
    assert_eq!(
        duplicate_code_string(&codes),
        None,
        "each enum must map its variants to a distinct string"
    );
}

/// Synthetic table rows where `INVALID_PARAMETERS` appears only in a
/// description.
const ROWS_WITHOUT_CODE_ROW: [&str; 3] = [
    "| Code variant | `as_str()` value | Producing condition |",
    "| --- | --- | --- |",
    "| `EmptyBuild` | `EMPTY_BUILD` | described as `INVALID_PARAMETERS` |",
];

/// Synthetic table rows where `INVALID_PARAMETERS` fills its own value cell.
const ROWS_WITH_CODE_ROW: [&str; 3] = [
    "| Code variant | `as_str()` value | Producing condition |",
    "| --- | --- | --- |",
    "| `InvalidParameters` | `INVALID_PARAMETERS` | described as `EMPTY_BUILD` |",
];

/// Verifies that a description mention does not stand in for a code's row.
///
/// This is the regression case: the token appears in another row's text, so
/// the check must still report the code as missing until it owns an
/// `as_str()` cell.
#[test]
fn a_description_mention_does_not_satisfy_the_row_check() {
    assert_eq!(
        missing_code(&ROWS_WITHOUT_CODE_ROW, &["INVALID_PARAMETERS"]),
        Some("INVALID_PARAMETERS")
    );
}

/// Verifies that a code in its own `as_str()` cell satisfies the row check.
#[test]
fn an_as_str_cell_satisfies_the_row_check() {
    assert_eq!(
        missing_code(&ROWS_WITH_CODE_ROW, &["INVALID_PARAMETERS"]),
        None
    );
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
