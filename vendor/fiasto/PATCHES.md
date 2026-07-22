# Patches relative to fiasto 0.2.7

Upstream: https://github.com/alexhallam/fiasto (crates.io `fiasto` 0.2.7).

## Panic-free term / family / response fallbacks

`parse_term`, `parse_family`, and `parse_response` used `unreachable!()` on match
arms that malformed Wilkinson input can reach (for example `Rey ~ 1 (+ (1 |`,
where `1(` is accepted as a term start and then treated as a function call).

Those arms now return `ParseError::Unexpected` so formula parsing never aborts
the process. This matters for libFuzzer / cargo-fuzz builds that force
`panic=abort`, where `catch_unwind` cannot contain the old panic.

## Publishing note

`lme-rs` depends on this path crate. A crates.io release of `lme-rs` will need
either an upstream `fiasto` release with the same fix, or a published fork
replacing the path dependency.
