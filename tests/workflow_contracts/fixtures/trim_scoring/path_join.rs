// A path join, which is not a dispatch: `Path::join` takes a path, and the
// single argument is what stops the receiver form reporting it as a pool join.

fn score_trim_jobs(jobs: Vec<Job>) -> PathBuf {
    cache_dir.join(&cache_file)
}
