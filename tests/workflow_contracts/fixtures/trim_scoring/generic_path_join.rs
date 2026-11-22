// A path join whose single argument is a call with its own type arguments.
// The comma between `A` and `B` separates type arguments, not values, so this
// is one argument; a scan that counted every comma at the list's level read
// the body as a pool join and failed a correct scorer.

fn score_trim_jobs(jobs: Vec<Job>) -> PathBuf {
    cache_dir.join(make_path::<A, B>())
}
