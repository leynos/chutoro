// A dispatch in a sibling function that the pinned list does not name. A
// file-level search would report it and fail a scorer that is itself serial.

fn score_trim_jobs(jobs: Vec<Job>) -> Vec<Scored> {
    jobs.into_iter().map(score).collect()
}

fn build_worker_rngs(jobs: Vec<Job>) -> Vec<Scored> {
    jobs.into_par_iter().map(score).collect()
}
