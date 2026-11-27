//! Unit tests for output-root precedence and metadata parsing.

use std::path::{Path, PathBuf};

use super::{OutputRootInputs, parse_metadata, resolve_output_root};

fn resolve(
    criterion_home: Option<&Path>,
    cargo_target_dir: Option<&Path>,
    metadata_target_dir: Option<&Path>,
) -> PathBuf {
    resolve_output_root(&OutputRootInputs {
        criterion_home,
        cargo_target_dir,
        metadata_target_dir,
        working_directory: Path::new("/work"),
    })
}

#[test]
fn criterion_home_wins_over_every_other_source() {
    let root = resolve(
        Some(Path::new("/explicit")),
        Some(Path::new("/target")),
        Some(Path::new("/metadata")),
    );

    assert_eq!(root, Path::new("/explicit"));
}

#[test]
fn relative_criterion_home_is_resolved_against_the_working_directory() {
    let root = resolve(Some(Path::new("custom/criterion")), None, None);

    assert_eq!(root, Path::new("/work/custom/criterion"));
}

#[test]
fn cargo_target_directory_wins_over_metadata() {
    let root = resolve(
        None,
        Some(Path::new("/cargo-target")),
        Some(Path::new("/metadata")),
    );

    assert_eq!(root, Path::new("/cargo-target/criterion"));
}

#[test]
fn relative_cargo_target_directory_is_resolved_against_the_working_directory() {
    let root = resolve(None, Some(Path::new("build")), None);

    assert_eq!(root, Path::new("/work/build/criterion"));
}

#[test]
fn metadata_target_directory_is_used_when_no_override_exists() {
    let root = resolve(None, None, Some(Path::new("/metadata")));

    assert_eq!(root, Path::new("/metadata/criterion"));
}

#[test]
fn default_root_matches_criterion_last_resort() {
    let root = resolve(None, None, None);

    assert_eq!(root, Path::new("/work/target/criterion"));
}

#[test]
fn malformed_metadata_names_the_workspace() {
    let error = parse_metadata(b"{not json", Path::new("/reference"))
        .expect_err("malformed metadata must fail");

    let message = error.to_string();
    assert!(message.contains("/reference"), "{message}");
    assert!(message.contains("cargo metadata"), "{message}");
}

#[test]
fn metadata_without_a_target_directory_is_rejected() {
    let error = parse_metadata(br#"{"packages":[]}"#, Path::new("/reference"))
        .expect_err("missing target directory must fail");

    assert!(error.to_string().contains("target_directory"), "{error}");
}
