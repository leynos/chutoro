// A slice join in the scored body. `join(` is a Rayon free function, but
// `names.join(", ")` is `[String]::join`, which dispatches nothing; a
// substring matcher reported it and failed a correct body.

fn score_trim_jobs(jobs: Vec<Job>) -> String {
    names.join(", ")
}
