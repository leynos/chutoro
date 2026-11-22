// `par_char_indices` dispatches a string by character index. It is not
// `par_chars` followed by more text, so the character entry misses it.

fn score_trim_jobs(text: &str) -> usize {
    text.par_char_indices().map(|(index, _)| index).sum()
}
