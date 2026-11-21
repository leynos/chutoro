// A path join whose single argument is generic in its own right. The comma
// between `A` and `B` sits inside `Outer<A, B>`, so only the comma after the
// inner list is at the outer list's level and even that one belongs to the
// turbofish; a skip that stopped at the first `>` ended inside the inner list
// and read the body as a pool join.

fn score_trim_jobs(jobs: Vec<Job>) -> PathBuf {
    cache_dir.join(make_path::<Outer<A, B>, C>())
}
