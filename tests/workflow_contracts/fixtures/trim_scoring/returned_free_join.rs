// A free call written after a keyword. `return join(left, right)` dispatches
// exactly as `join(left, right)` does, but the word before the name ends in an
// identifier character, so a reading that treated every preceding identifier as
// a receiver reported this body as serial.

fn score_trim_jobs(jobs: Vec<Job>) -> Scored {
    return join(|| score(left), || score(right));
}
