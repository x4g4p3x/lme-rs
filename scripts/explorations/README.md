# Standalone explorations

These are **library probes**, not CI tooling. They replace the old fiasto AST scripts that were removed with the native Wilkinson parser.

Run from the repository root (`--release` is optional for the AST dump; use it for the θ-grid and MCP fits):

```bash
cargo run --locked --example explore_formula_ast
cargo run --release --locked --example explore_theta_grid
cargo run --release --locked --example explore_mcp_adjust
task explorations
```

| Example | What it probes |
|:--------|:---------------|
| [`examples/explore_formula_ast.rs`](../../examples/explore_formula_ast.rs) | Native parser dump: intercept, generated columns, `||` vs `|`, nested `batch/cask`, `poly` / `ns`, `I()`, offsets |
| [`examples/explore_theta_grid.rs`](../../examples/explore_theta_grid.rs) | 1-D REML θ grid on intercept-only sleepstudy vs the MLE from [`lmer`](../../src/lib.rs) |
| [`examples/explore_mcp_adjust.rs`](../../examples/explore_mcp_adjust.rs) | Pastes `cask` Tukey family: raw vs Bonferroni vs Holm vs Tukey–Kramer p-values |

Invariants are locked by [`tests/test_explorations.rs`](../../tests/test_explorations.rs) (also in the consolidated integration harness). `task explorations` / `python3 scripts/ci/lme_ci.py explorations` runs the three examples.
