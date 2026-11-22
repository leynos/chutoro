// A sequential binding whose name matches an iterator marker. A body that
// only binds `par_iter` performs no parallel call, so matching the
// identifier alone would fail a correct scorer.

fn score_trim_jobs(jobs: Vec<Job>) -> usize {
    let par_iter = jobs.iter();
    par_iter.count()
}
