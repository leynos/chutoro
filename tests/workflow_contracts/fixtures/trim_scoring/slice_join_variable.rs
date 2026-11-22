// A slice join whose separator is a variable. Here no comma follows the
// argument at all, which is what the receiver form requires, so a separator
// assembled at run time stays a slice join rather than a pool join.

fn score_trim_jobs(jobs: Vec<Job>) -> String {
    names.join(separator)
}
