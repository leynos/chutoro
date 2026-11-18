//! Compile-pass fixture for the opaque cache and object key surface.

use chutoro_bench_datasets::{CacheKey, ObjectKey};

fn main() {
    let cache: CacheKey = CacheKey::new("cache/a.bin");
    let owned_cache = CacheKey::new(String::from("cache/b.bin"));
    let object: ObjectKey = ObjectKey::new("prepared/a.bin");
    let owned_object = ObjectKey::new(String::from("prepared/b.bin"));

    assert_eq!(cache.as_ref(), "cache/a.bin");
    assert_eq!(owned_cache.to_string(), "cache/b.bin");
    assert_eq!(format!("{object}"), "prepared/a.bin");
    assert_eq!(owned_object.as_ref(), "prepared/b.bin");
}