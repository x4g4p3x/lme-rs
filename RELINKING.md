# Relinking and source information

Release binaries include `sprs-ldl` 0.10.0, an LGPL-2.1 component. The corresponding source for the application and its locked Rust dependency graph is available in each matching `lme-rs` source release.

To rebuild a binary with a modified `sprs-ldl` implementation:

```text
git clone https://github.com/x4g4p3x/lme-rs.git
cd lme-rs
git checkout <the release tag used for the binary>
cargo build --release --locked
```

For Python artifacts, build the corresponding wheel from that same source tree:

```text
cd python
python -m pip install maturin
maturin build --release --locked
```

The relevant license text is [`LICENSES/LGPL-2.1-only.txt`](LICENSES/LGPL-2.1-only.txt). Distributors who modify, statically link, or redistribute artifacts must meet the applicable LGPL source, notice, and relinking obligations for their distribution method.
