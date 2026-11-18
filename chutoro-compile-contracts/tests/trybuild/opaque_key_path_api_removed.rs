//! Compile-fail fixture proving cache and object keys expose no path API.

use camino::Utf8Path;
use chutoro_bench_datasets::{CacheKey, ObjectKey};

fn main() {
    let cache = CacheKey::new("cache/a.bin");
    let _ = cache.to_path_buf();
    let _ = cache.clone().into_inner();
    let _: &Utf8Path = cache.as_ref();

    let object = ObjectKey::new("prepared/a.bin");
    let _ = object.to_path_buf();
    let _ = object.clone().into_inner();
    let _: &Utf8Path = object.as_ref();
}