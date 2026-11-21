// An associated-function call to a pool entry point. Rust lets an inherent
// method be called through its type, so `ThreadPool::scope(pool, ...)` runs
// the closure on the pool exactly as `pool.scope(...)` does. The free-function
// anchor rejects the `::` and the receiver pattern requires a dot, so neither
// entry read this dispatch.

fn score_trim_jobs(jobs: Vec<Job>) {
    ThreadPool::scope(pool, |_| score(jobs));
}
