// A path join written with whitespace around the dot. Rust permits it, so
// `cache_dir . join(x)` is the same method call as `cache_dir.join(x)`; a
// free-function reading that only required the name to start a word reported
// the body as a pool join, because the character before the name is a space.

fn score_trim_jobs(jobs: Vec<Job>) -> PathBuf {
    cache_dir . join(&cache_file)
}
