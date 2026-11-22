// An in-place scope call. Unlike `scope`, `in_place_scope` runs the closure
// on the calling thread; only work the closure spawns through the scope runs
// on the pool, and the call waits for that work. The dispatch is still
// shared-pool work, and the longer name matches neither the `scope` receiver
// pattern nor its free-call anchor.

fn score_trim_jobs(jobs: Vec<Job>) {
    pool.in_place_scope(|scope| run(scope));
}
