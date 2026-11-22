// An associated-function call whose entry point carries a turbofish. The type
// list sits between the name and the argument list, so a pattern expecting the
// parenthesis straight after the name read this dispatch as serial.

fn score_trim_jobs(jobs: Vec<Job>) {
    ThreadPool::install::<_, ()>(pool, || score(jobs));
}
