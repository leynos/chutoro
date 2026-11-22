// A receiver-form call whose entry point carries a turbofish. `ThreadPool`'s
// `scope` is generic in its operation and its result, so the type list may sit
// between the name and the argument list; a pattern expecting the parenthesis
// straight after the name read this dispatch as serial.

fn score_trim_jobs(jobs: Vec<Job>) {
    pool.scope::<_, ()>(|_| score(jobs));
}
