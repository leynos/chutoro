// A pool join whose turbofish holds tuple types. Both results of
// `ThreadPool::join` are spelled as parenthesized types here, so the call
// never reached the argument-list check and the body read as a slice join.

fn score_trim_jobs(jobs: Vec<Job>) -> (Scored, Scored) {
    pool.join::<_, _, (), ()>(|| score(left), || score(right))
}
