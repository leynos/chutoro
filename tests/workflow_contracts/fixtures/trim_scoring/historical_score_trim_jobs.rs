// The former scorer, copied verbatim from commit
// 56d6909c7019ac7caa6d485d11ebf5708b47528e,
// ``chutoro-core/src/hnsw/cpu/trim.rs`` lines 35-48.
//
// It is checked in rather than read from Git history so the contract keeps
// its evidence when history is shallow. The marker it carries is the
// regression the contract exists to reject.

    pub(crate) fn score_trim_jobs<D: DataSource + Sync>(
        &self,
        trim_jobs: Vec<TrimJob>,
        source: &D,
    ) -> Result<Vec<TrimResult>, HnswError> {
        if trim_jobs.is_empty() {
            return Ok(Vec::new());
        }

        trim_jobs
            .into_par_iter()
            .map(|job| self.run_trim_job(job, source))
            .collect::<Result<Vec<_>, HnswError>>()
    }
