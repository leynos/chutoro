//! Integration tests for crate-provided testing adapters.

use bytes::Bytes;
use camino::Utf8PathBuf;
use cap_std::fs_utf8::Dir;
use chutoro_bench_datasets::{
    CacheKey, FetchSizeExceeded, Fetcher, RecipeError, SourceUrl, Storage,
    testing::{FilesystemFetcher, InMemoryStorage},
};
use rstest::rstest;

#[test]
fn in_memory_storage_gets_and_overwrites_records() {
    let storage = InMemoryStorage::default();
    let key = CacheKey::new("cache/example.bin");

    assert_eq!(
        storage.get(&key).expect("storage read should succeed"),
        None
    );
    storage
        .put(&key, b"first")
        .expect("storage write should succeed");
    assert_eq!(
        storage.get(&key).expect("storage read should succeed"),
        Some(Bytes::from_static(b"first"))
    );
    storage
        .put(&key, b"second")
        .expect("storage overwrite should succeed");
    assert_eq!(
        storage.get(&key).expect("storage read should succeed"),
        Some(Bytes::from_static(b"second"))
    );
}

#[rstest]
#[case::unsupported_scheme("https://example.test/data.bin")]
#[case::parent_directory("file://../outside.bin")]
#[case::absolute_path("file:///outside.bin")]
fn filesystem_fetcher_rejects_invalid_file_sources(#[case] value: &str) {
    let root = tempfile::tempdir().expect("temporary fixture root should be created");
    let fetcher = FilesystemFetcher::new(
        utf8_path_buf(root.path()).expect("temporary path should be valid UTF-8"),
    );
    let source = SourceUrl::parse(value).expect("invalid-source test URL should parse");
    let error = fetcher
        .fetch_bytes(&source, 1024)
        .expect_err("invalid file source should fail");

    assert!(matches!(error, RecipeError::InvalidSource(_)));
}

#[test]
fn filesystem_fetcher_enforces_size_limit_during_read() {
    let root = tempfile::tempdir().expect("temporary fixture root should be created");
    let root_path = utf8_path_buf(root.path()).expect("temporary path should be valid UTF-8");
    let fixture_dir = Dir::open_ambient_dir(&root_path, cap_std::ambient_authority())
        .expect("fixture directory should open");
    fixture_dir
        .write("dataset.bin", b"abcd")
        .expect("fixture file should be writable");
    let fetcher = FilesystemFetcher::new(root_path);
    let source = SourceUrl::parse("file://dataset.bin").expect("fixture source URL should parse");
    let error = fetcher
        .fetch_bytes(&source, 3)
        .expect_err("oversized fixture file should fail");

    assert!(matches!(
        error,
        RecipeError::FetchSizeExceeded(FetchSizeExceeded { limit_bytes: 3, .. })
    ));
}

fn utf8_path_buf(path: &std::path::Path) -> Option<Utf8PathBuf> {
    Utf8PathBuf::from_path_buf(path.to_path_buf()).ok()
}
