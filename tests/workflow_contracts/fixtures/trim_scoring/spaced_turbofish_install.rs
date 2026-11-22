// A method call whose turbofish is spaced. Rust permits whitespace between
// `::` and `<`, so `pool.install:: <_, ()>(...)` is the same call; a pattern
// that required them adjacent read this dispatch as serial.

fn score_trim_jobs(jobs: Vec<Job>) {
    pool.install:: <_, ()>(|| score(jobs));
}
