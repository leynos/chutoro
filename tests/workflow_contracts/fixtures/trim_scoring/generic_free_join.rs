// A free call to a generic entry point. Rayon's free functions are generic, so
// `join::<_, _, (), ()>(left, right)` dispatches exactly as `join(left, right)`
// does; a pattern that expected the parenthesis straight after the name read
// this body as serial.

fn score_trim_jobs(jobs: Vec<Job>) -> Scored {
    join::<_, _, (), ()>(|| score(left), || score(right))
}
