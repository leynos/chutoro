// A pool install whose type arguments include a const expression. The `1 << 2`
// in `[u8; { 1 << 2 }]` spells two angle brackets that open no type list, so a
// scan that counted them ran past the call and read this dispatch as serial.

fn score_trim_jobs(jobs: Vec<Job>) {
    pool.install::<_, [u8; { 1 << 2 }]>(|| score(jobs));
}
