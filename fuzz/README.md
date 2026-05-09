# LibFuzzer targets for `lme-rs`

This directory follows the [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) layout: LLVM **libFuzzer** drives coverage-guided mutation of arbitrary byte inputs, which are turned into formula strings and exercised against the real parser and (optionally) design-matrix construction.

## Prerequisites

1. **Nightly Rust** for the `fuzz/` workspace (see `rust-toolchain.toml` in this directory).
2. **cargo-fuzz**: `cargo install cargo-fuzz`

## Targets

| Target               | What it exercises |
|----------------------|-------------------|
| `formula_parse`      | `lme_rs::formula::parse` only (fiasto + nested/`||` expansion + offset stripping). |
| `formula_pipeline`   | Successful parses additionally run `build_design_matrices` on a small synthetic `DataFrame` built from AST column names. |

## Build

```bash
cd fuzz
cargo fuzz build formula_parse
cargo fuzz build formula_pipeline
```

## Run (with checked-in seed corpus)

```bash
cd fuzz
cargo fuzz run formula_parse corpus/formula_parse -- -runs=100000 -max_len=32768
cargo fuzz run formula_pipeline corpus/formula_parse -- -runs=50000 -max_len=32768
```

### Windows and AddressSanitizer

The default sanitizer is **AddressSanitizer**. On some Windows setups the fuzzer binary fails at launch with **STATUS_DLL_NOT_FOUND** (missing ASAN runtime). In that case use an unsanitized build for local smoke runs:

```bash
cd fuzz
cargo fuzz run formula_parse corpus/formula_parse --sanitizer none -- -runs=20000 -max_len=8192
```

Linux and macOS typically run the default ASAN build without extra setup.

## Artifacts

Crashes and hangs are written under `fuzz/artifacts/<target>/` (gitignored). Reproduce with:

```bash
cargo fuzz run formula_parse path/to/crash-file
```

## CI

The repository workflow **Fuzz (manual)** runs a short libFuzzer smoke job from the Actions tab (`workflow_dispatch`).
