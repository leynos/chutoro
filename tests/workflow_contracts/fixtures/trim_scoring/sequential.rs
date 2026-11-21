// A body that scores in place: the shape the contract requires.

fn score_trim_jobs(jobs: Vec<Job>) -> Vec<Scored> {
    jobs.into_iter().map(score).collect()
}
