// `par_bytes` dispatches a string's bytes, a method no other entry names.

fn score_trim_jobs(text: &str) -> usize {
    text.par_bytes().map(|byte| byte as usize).sum()
}
