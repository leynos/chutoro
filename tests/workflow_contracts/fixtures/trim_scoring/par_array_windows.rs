// `par_array_windows` dispatches constant-size windows. It is not spelled
// with `par_windows`, so the window entry misses it.

fn score_trim_jobs(jobs: &[Job]) -> usize {
    jobs.par_array_windows::<2>().count()
}
