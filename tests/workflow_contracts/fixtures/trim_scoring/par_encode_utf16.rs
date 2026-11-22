// `par_encode_utf16` dispatches a string's UTF-16 encoding.

fn score_trim_jobs(text: &str) -> usize {
    text.par_encode_utf16().count()
}
