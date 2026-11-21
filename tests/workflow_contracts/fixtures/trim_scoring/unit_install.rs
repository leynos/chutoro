// A pool install whose turbofish holds a unit type. `ThreadPool::install` is
// generic in its closure and its result, and a result of `()` needs
// parentheses inside the turbofish; a pattern that ruled parentheses out did
// not see the call at all, so this dispatch read as serial.

fn score_trim_jobs(jobs: Vec<Job>) {
    pool.install::<_, ()>(|| score(jobs))
}
