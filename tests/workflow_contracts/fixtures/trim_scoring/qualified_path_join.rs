// A path join whose single argument is a qualified path. The comma in
// `<Paths<A, B> as HasPath>::path()` separates type arguments rather than
// values, so this is one argument; a scan that read it as two reported a
// parallel join and failed a correct scorer.

fn score_trim_jobs(jobs: Vec<Job>) -> PathBuf {
    cache_dir.join(<Paths<A, B> as HasPath>::path())
}
