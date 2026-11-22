// `par_matches` dispatches a string's matches of a pattern.

fn score_trim_jobs(text: &str) -> usize {
    text.par_matches('a').count()
}
