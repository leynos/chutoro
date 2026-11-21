// A local of unit and tuple type, which dispatches nothing. Parenthesised
// types are ordinary Rust, so recognising them inside a turbofish must not
// turn every tuple in a body into a call.

fn score_trim_jobs(jobs: Vec<Job>) -> () {
    let pair: (u8, u8) = (1, 2);
    ()
}
