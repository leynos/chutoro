// A pool join, which is a dispatch: `ThreadPool::join` runs both closures on
// the pool, unlike the slice join above. The receiver form needed its own
// marker, and the two-argument shape is what keeps the two apart.

fn score_trim_jobs(jobs: Vec<Job>) -> Scored {
    pool.join(|| score(left), || score(right))
}
