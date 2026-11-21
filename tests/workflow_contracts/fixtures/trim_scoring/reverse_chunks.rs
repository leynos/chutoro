// A reverse-chunk iterator, which dispatches as surely as `par_chunks`.
// `par_rchunks` does not contain `par_chunks`, so the ordinary chunk marker
// missed this body.

fn score_trim_jobs(jobs: Vec<Job>) -> Vec<Scored> {
    jobs.par_rchunks(2).map(score).collect()
}
