// A pool join of bound callables. `ThreadPool::join` accepts anything that
// implements `FnOnce`, so `pool.join(left, right)` dispatches exactly as the
// inline-closure form does; a pattern that required `|` after the parenthesis
// read this body as serial.

fn score_trim_jobs(jobs: Vec<Job>) -> Scored {
    pool.join(score_left, score_right)
}
