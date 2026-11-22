// A free call through a grouped operand. Rust permits a parenthesised
// expression as the callee, so `(join)(left, right)` dispatches exactly as
// `join(left, right)` does; a reading that required the argument list straight
// after the name read this body as serial.

fn score_trim_jobs(jobs: Vec<Job>) -> Scored {
    (join)(|| score(left), || score(right))
}
