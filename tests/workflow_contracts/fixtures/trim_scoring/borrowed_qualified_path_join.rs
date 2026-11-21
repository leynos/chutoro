// A borrowed qualified path as a join's single argument. The `&` begins the
// value rather than ending an operand, so the angle brackets still hold type
// arguments and the call passes one argument.

fn score_trim_jobs(jobs: Vec<Job>) -> PathBuf {
    cache_dir.join(&<Paths<A, B> as HasPath>::path())
}
