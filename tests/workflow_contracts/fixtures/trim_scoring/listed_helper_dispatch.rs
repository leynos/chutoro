// A dispatch in a helper the pinned list does name. Scoping the reading to
// named bodies must not become a way to miss a listed helper: `run_trim_job`
// computes the distances a trim job needs, so dispatch there reaches the pool
// from inside the critical section just as surely as dispatch in the scorer.

fn score_trim_jobs(jobs: Vec<Job>) -> Vec<Scored> {
    jobs.into_iter().map(|job| run_trim_job(job)).collect()
}

fn run_trim_job(job: Job) -> Scored {
    batch.into_par_iter().map(score).collect()
}
