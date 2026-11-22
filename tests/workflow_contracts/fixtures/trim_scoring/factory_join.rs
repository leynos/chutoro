// A pool join whose first closure is built by a call. `ThreadPool::join`
// takes any two `FnOnce` values, so a factory call returning one dispatches
// like any other operand; a pattern that allowed only a closure or a bare
// name as the first argument read this body as serial.

fn score_trim_jobs(jobs: Vec<Job>) -> Scored {
    pool.join(make_job(), || ())
}
