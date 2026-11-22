// The former regression, restored. This is the shape the contract exists to
// reject, and the shape a benchmark once spent twenty-five minutes inside.

fn score_trim_jobs(jobs: Vec<Job>) -> Vec<Scored> {
    jobs.into_par_iter().map(score).collect()
}
