//! LibFuzzer harness: arbitrary bytes → lossy UTF-8 → `formula::parse`.
//!
//! Run (requires nightly + `cargo install cargo-fuzz`):
//! `cd fuzz && cargo +nightly fuzz run formula_parse`
//!
//! Corpus: `fuzz/corpus/formula_parse/` (checked-in seeds).

#![no_main]

use libfuzzer_sys::fuzz_target;

const MAX_INPUT: usize = 32 * 1024;

fuzz_target!(|data: &[u8]| {
    let slice = if data.len() > MAX_INPUT {
        &data[..MAX_INPUT]
    } else {
        data
    };
    let s = String::from_utf8_lossy(slice);
    let _ = lme_rs::formula::parse(s.as_ref());
});
