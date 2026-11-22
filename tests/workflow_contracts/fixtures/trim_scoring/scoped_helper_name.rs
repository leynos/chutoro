// A helper whose name ends in `scope`. A substring match on `scope(` read it
// as Rayon's `scope`; the boundary anchor leaves it alone.

fn score_trim_jobs(jobs: Vec<Job>) -> Vec<Scored> {
    with_scope(|| jobs.into_iter().map(score).collect())
}
