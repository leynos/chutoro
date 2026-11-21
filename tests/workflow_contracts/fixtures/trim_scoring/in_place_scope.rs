// An in-place scope call. `in_place_scope` runs its closure on the shared
// pool and waits for it, exactly as `scope` does, but the longer name matches
// neither the `scope` receiver pattern nor its free-call anchor.

fn score_trim_jobs(jobs: Vec<Job>) {
    pool.in_place_scope(|scope| run(scope));
}
