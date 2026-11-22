// A slice join written with a trailing comma. Rust permits the comma after a
// single argument, so a marker that read any comma as a second argument
// reported this correct body as a pool join.

fn score_trim_jobs(jobs: Vec<Job>) -> String {
    names.join(separator,)
}
