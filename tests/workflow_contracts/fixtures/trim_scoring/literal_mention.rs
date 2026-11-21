// The same mention inside a string literal, which a reader that blanked only
// comments would still report. A warning about the dispatch is not one.

fn score_trim_jobs(jobs: Vec<Job>) -> Vec<Scored> {
    let warning = "into_par_iter() strands the insert mutex holder";
    jobs.into_iter().map(score).collect()
}
