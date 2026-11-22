// `par_match_indices` dispatches the positions of a string's matches. Its
// spelling is not a prefix of `par_matches`, so it needs its own entry.

fn score_trim_jobs(text: &str) -> usize {
    text.par_match_indices('a').count()
}
