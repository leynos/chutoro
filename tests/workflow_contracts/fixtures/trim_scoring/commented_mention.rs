// A comment in the scored body naming the dispatch it must not contain. A
// reading that scanned raw text would report this and fail a correct body,
// so the existing comment explaining the deadlock must stay harmless.

fn score_trim_jobs(jobs: Vec<Job>) -> Vec<Scored> {
    // into_par_iter() would strand the insert mutex holder.
    jobs.into_iter().map(score).collect()
}
