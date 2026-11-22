// `par_chunk_by` dispatches a slice by a predicate. It shares its prefix with
// neither `par_chunks` nor `par_rchunks`, so the chunk entries miss it.

fn score_trim_jobs(jobs: &[Job]) -> usize {
    jobs.par_chunk_by(|left, right| left == right).count()
}
