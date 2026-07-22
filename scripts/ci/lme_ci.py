#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Cross-platform CI runner for lme-rs — single source of truth for local + GitHub Actions."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
PYTHON_VENV = PYTHON_DIR / ".venv"
JULIA_FORMAT_SCRIPT = ROOT / "scripts" / "ci" / "julia_format.jl"
R_FORMAT_SCRIPT = ROOT / "scripts" / "ci" / "r_format.R"


class CiError(Exception):
    pass


def _echo(cmd: Sequence[str], *, cwd: Path | None = None) -> None:
    where = f" (cwd={cwd})" if cwd else ""
    print(f"==> {' '.join(cmd)}{where}", flush=True)


def run(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    _echo(cmd, cwd=cwd)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    result = subprocess.run(
        list(cmd),
        cwd=cwd or ROOT,
        env=merged,
        text=True,
    )
    if check and result.returncode != 0:
        raise CiError(f"command failed ({result.returncode}): {' '.join(cmd)}")
    return result


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise CiError(
            f"{name!r} not found on PATH. Install via mise (`mise install`) or see CONTRIBUTING.md."
        )
    return path


def venv_python(venv: Path = PYTHON_VENV) -> Path:
    win = venv / "Scripts" / "python.exe"
    if win.exists():
        return win
    unix = venv / "bin" / "python"
    if unix.exists():
        return unix
    raise CiError(f"no python executable in {venv}")


def staged_files(pattern: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    if not result.stdout:
        return []
    paths = [p for p in result.stdout.split("\0") if p]
    if pattern == "rust":
        return [p for p in paths if p.endswith(".rs") and not p.startswith("target/")]
    if pattern == "python":
        return [p for p in paths if p.startswith("python/") and p.endswith(".py")]
    if pattern == "comparison-r":
        return [
            p
            for p in paths
            if p.endswith(".R") and (p.startswith("comparisons/") or p.startswith("tests/"))
        ]
    if pattern == "comparison-jl":
        return [p for p in paths if p.endswith(".jl") and p.startswith("comparisons/")]
    return paths


def restage(paths: list[str]) -> None:
    if not paths:
        return
    run(["git", "add", "--", *paths], cwd=ROOT)


def cargo_fmt(*, apply: bool) -> None:
    cmd = ["cargo", "fmt", "--all"]
    if not apply:
        cmd.append("--check")
    run(cmd)


def cargo_clippy() -> None:
    run(["cargo", "clippy", "--locked", "--", "-D", "warnings"])


def cargo_check() -> None:
    run(["cargo", "check", "--workspace", "--all-targets", "--locked", "-v"])


def cargo_build_test() -> None:
    run(["cargo", "test", "--verbose", "--locked"])


def cargo_test_fast() -> None:
    """Unit tests only — skips integration/doc tests for quick feedback."""
    run(["cargo", "test", "--lib", "--locked"])


def cargo_doctest() -> None:
    run(
        ["cargo", "test", "--doc", "--locked", "--verbose"],
        env={"RUSTDOCFLAGS": "-D warnings"},
    )


def cargo_doc() -> None:
    run(
        ["cargo", "doc", "--no-deps", "--verbose", "--locked"],
        env={"RUSTDOCFLAGS": "-D warnings"},
    )


def cargo_audit() -> None:
    if not shutil.which("cargo-audit"):
        raise CiError(
            "cargo-audit not found on PATH. Install: cargo install cargo-audit "
            "(or see CONTRIBUTING.md / AGENTS.md)."
        )
    # `paste` is an unmaintained, build-time proc macro required by the current
    # argmin 0.11 release. RUSTSEC-2024-0436 is informational (no vulnerability
    # or patched argmin release). Keep this single exception narrow and deny all
    # other audit warnings so new advisories cannot silently accumulate.
    audit_cmd = [
        "cargo",
        "audit",
        "--deny",
        "warnings",
        "--ignore",
        "RUSTSEC-2024-0436",
    ]
    run(audit_cmd)
    if (PYTHON_DIR / "Cargo.toml").exists():
        run(audit_cmd, cwd=PYTHON_DIR)


def pip_audit() -> None:
    _require_tool("uv")
    _uv_sync(python="3.11")
    run(["uv", "run", "--no-sync", "pip-audit"], cwd=PYTHON_DIR)


def audit() -> None:
    """Security audit mirror of .github/workflows/audit.yml."""
    cargo_audit()
    pip_audit()


def repo_metadata_dry_run() -> None:
    run([sys.executable, "scripts/sync_github_repo_metadata.py", "--dry-run"])


def repo_metadata_verify() -> None:
    token = os.environ.get("REPO_ADMIN_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print(
            "skip: REPO_ADMIN_TOKEN not set (set locally to catch expired tokens before push)",
            flush=True,
        )
        return
    run([sys.executable, "scripts/sync_github_repo_metadata.py", "--verify-token"])


def repo_metadata() -> None:
    """Validate Cargo.toml-derived metadata; verify token when available."""
    repo_metadata_dry_run()
    repo_metadata_verify()


def completion_check() -> None:
    """Validate completion-manifest arithmetic and published score markers."""
    run([sys.executable, "scripts/ci/check_completion_score.py"])


def legal_compliance() -> None:
    """Validate fixture provenance and third-party license records."""
    run([sys.executable, "scripts/ci/check_legal_compliance.py"])


def benchmarks_smoke() -> None:
    """Fast Rust-only cross-language benchmark smoke (release examples)."""
    run(["cargo", "build", "--release", "--locked", "--examples"])
    run(
        [
            sys.executable,
            "scripts/run_cross_language_benchmarks.py",
            "--cases",
            "sleepstudy",
            "--implementations",
            "rust",
            "--warmups",
            "0",
            "--repeats",
            "1",
            "--skip-rust-build",
            "--output",
            "benchmark-results/preflight-cross-language.json",
        ]
    )


def benchmarks_r_smoke() -> None:
    """Run one R comparison script when Rscript and lme4 are available."""
    if not shutil.which("Rscript"):
        print("skip: Rscript not installed (full R benchmarks are CI-only)", flush=True)
        return
    probe = subprocess.run(
        ["Rscript", "-e", "suppressPackageStartupMessages(library(lme4))"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        print("skip: R package lme4 not installed (full R benchmarks are CI-only)", flush=True)
        return
    run(["Rscript", "comparisons/sleepstudy.R"])


def benchmarks_preflight() -> None:
    benchmarks_smoke()
    benchmarks_r_smoke()


def _resolve_julia_bin() -> str | None:
    env = os.environ.get("JULIA_BIN")
    if env and Path(env).exists():
        return env
    found = shutil.which("julia")
    if found:
        return found
    if os.name == "nt":
        programs = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs"
        if programs.is_dir():
            matches = sorted(programs.glob("Julia-*/bin/julia.exe"))
            if matches:
                return str(matches[-1])
    return None


def perf_breakdown() -> None:
    """Rust LME_PERF_DIAG vs Julia optsum.feval on fair fixtures."""
    run([sys.executable, "scripts/run_perf_breakdown.py", "--cases", "crossed_20k"])


def benchmarks_fair_rust_julia() -> None:
    """Fair fit-only Rust vs Julia timing (optional when Julia + MixedModels are installed)."""
    julia = _resolve_julia_bin()
    if not julia:
        print(
            "skip: julia not found (set PATH or JULIA_BIN for fair Rust/Julia benchmarks)",
            flush=True,
        )
        return
    probe = subprocess.run(
        [julia, "-e", "using CSV, DataFrames, JSON, MixedModels"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        stderr = (probe.stderr or probe.stdout or "").strip()
        print(
            "skip: Julia packages CSV/DataFrames/JSON/MixedModels not available "
            f"({stderr or 'Pkg.add missing packages'})",
            flush=True,
        )
        return
    run(
        [
            sys.executable,
            "scripts/run_fair_rust_julia_benchmark.py",
            "--cases",
            "sleepstudy_reml,random_intercept_10k",
            "--warmups",
            "1",
            "--repeats",
            "2",
            "--julia",
            julia,
        ]
    )


def print_benchmark_failures(path: str) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for failure in payload.get("failures", []):
        print(
            f"- {failure['case']} [{failure['implementation']}]: {failure.get('error')}",
            file=sys.stderr,
        )
        details = failure.get("details")
        if details:
            print(f"  {details}", file=sys.stderr)


def preflight() -> None:
    """Pre-push gate: static checks + compile graph + security audit."""
    lint()
    cargo_check()
    cargo_audit()
    legal_compliance()
    repo_metadata_dry_run()
    repo_metadata_verify()
    print(
        "lme_ci.py preflight: OK (lint + check + cargo audit + legal + repo metadata). "
        "Run `task ci` before large changes; macOS aarch64 BLAS is CI-only.",
        flush=True,
    )


def rust_lint() -> None:
    cargo_fmt(apply=False)
    cargo_clippy()


def _ruff_invocation() -> tuple[list[str], str]:
    _require_tool("uv")
    return ["uv", "tool", "run", "ruff"], str(PYTHON_DIR / "pyproject.toml")


# Linted on `task lint` / pre-push (excludes .venv via pyproject exclude).
RUFF_PATHS = [PYTHON_DIR / "tests", PYTHON_DIR / "examples"]


def ruff_lint() -> None:
    cmd, config = _ruff_invocation()
    paths = [str(p.relative_to(ROOT)) for p in RUFF_PATHS]
    run([*cmd, "check", "--config", config, *paths])
    run([*cmd, "format", "--check", "--config", config, *paths])


def lint() -> None:
    """Static checks for Rust and Python (no tests, no builds)."""
    rust_lint()
    ruff_lint()


def rust_all() -> None:
    rust_lint()
    cargo_check()
    cargo_build_test()
    cargo_doctest()
    cargo_doc()


def ruff_staged(*, fix: bool) -> None:
    files = staged_files("python")
    if not files:
        return
    cmd, config = _ruff_invocation()
    if fix:
        run([*cmd, "check", "--fix", "--config", config, *files])
        run([*cmd, "format", "--config", config, *files])
        restage(files)
    else:
        run([*cmd, "check", "--config", config, *files])
        run([*cmd, "format", "--check", "--config", config, *files])


def comparison_r_files() -> list[str]:
    paths = sorted((ROOT / "comparisons").rglob("*.R"))
    paths.extend(sorted((ROOT / "tests").glob("*.R")))
    return [str(p.relative_to(ROOT)).replace("\\", "/") for p in paths]


def comparison_jl_files() -> list[str]:
    paths = sorted((ROOT / "comparisons").rglob("*.jl"))
    return [str(p.relative_to(ROOT)).replace("\\", "/") for p in paths]


def _r_styler_ready(*, required: bool) -> bool:
    if not shutil.which("Rscript"):
        message = "skip: Rscript not installed (comparison R formatting is optional locally)"
        if required:
            raise CiError("Rscript not found on PATH")
        print(message, flush=True)
        return False
    probe = subprocess.run(
        ["Rscript", "-e", "suppressPackageStartupMessages(library(styler))"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        message = (
            "skip: R package styler not installed "
            "(install.packages('styler') for comparison R formatting)"
        )
        if required:
            raise CiError("R package styler is not installed")
        print(message, flush=True)
        return False
    return True


def _julia_formatter_ready(*, required: bool) -> bool:
    if not shutil.which("julia"):
        message = "skip: julia not installed (comparison Julia formatting is optional locally)"
        if required:
            raise CiError("julia not found on PATH")
        print(message, flush=True)
        return False
    probe = subprocess.run(
        ["julia", "-e", "using JuliaFormatter"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        message = (
            "skip: Julia package JuliaFormatter not installed "
            "(Pkg.add(\"JuliaFormatter\") for comparison Julia formatting)"
        )
        if required:
            raise CiError("Julia package JuliaFormatter is not installed")
        print(message, flush=True)
        return False
    return True


def _format_r_files(files: list[str], *, check: bool, required: bool = False) -> None:
    if not files:
        return
    if not _r_styler_ready(required=required):
        return
    flag = "--check" if check else ""
    cmd = ["Rscript", str(R_FORMAT_SCRIPT.relative_to(ROOT))]
    if flag:
        cmd.append(flag)
    cmd.extend(files)
    run(cmd)


def _format_julia_files(files: list[str], *, check: bool, required: bool = False) -> None:
    if not files:
        return
    if not _julia_formatter_ready(required=required):
        return
    flag = "--check" if check else ""
    cmd = ["julia", str(JULIA_FORMAT_SCRIPT.relative_to(ROOT))]
    if flag:
        cmd.append(flag)
    cmd.extend(files)
    run(cmd)


def comparison_format_check(*, required: bool = False) -> None:
    """Format-check comparison / golden-parity R and Julia scripts."""
    r_files = comparison_r_files()
    jl_files = comparison_jl_files()
    _format_r_files(r_files, check=True, required=required)
    _format_julia_files(jl_files, check=True, required=required)


def r_format_staged(*, fix: bool) -> None:
    files = staged_files("comparison-r")
    if not files:
        return
    _format_r_files(files, check=not fix, required=False)
    if fix:
        restage(files)


def julia_format_staged(*, fix: bool) -> None:
    files = staged_files("comparison-jl")
    if not files:
        return
    _format_julia_files(files, check=not fix, required=False)
    if fix:
        restage(files)


def _venv_python_version(venv: Path) -> tuple[int, int]:
    py = venv_python(venv)
    result = subprocess.run(
        [str(py), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        capture_output=True,
        text=True,
        check=True,
    )
    major, minor = result.stdout.strip().split(".")
    return int(major), int(minor)


def _parse_python_version(python: str) -> tuple[int, int]:
    major, minor = python.split(".", 1)
    return int(major), int(minor)


def _uv_python_env() -> dict[str, str]:
    venv = PYTHON_VENV
    return {
        "PYO3_PYTHON": str(venv_python(venv)),
        "VIRTUAL_ENV": str(venv),
    }


def _uv_sync(*, python: str = "3.11", reuse: bool = True) -> None:
    """Install locked dev dependencies into python/.venv (uv-native flow)."""
    _require_tool("uv")
    want = _parse_python_version(python)
    if PYTHON_VENV.exists():
        if not reuse:
            shutil.rmtree(PYTHON_VENV)
        elif want != _venv_python_version(PYTHON_VENV):
            print(
                f"    (replacing python/.venv: need Python {python})",
                flush=True,
            )
            shutil.rmtree(PYTHON_VENV)
    run(
        [
            "uv",
            "sync",
            "--extra",
            "dev",
            "--python",
            python,
            "--no-install-project",
        ],
        cwd=PYTHON_DIR,
    )


def python_bindings(*, reuse_venv: bool = False, skip_wheel: bool = False) -> None:
    _uv_sync(python="3.11", reuse=reuse_venv)
    env = _uv_python_env()
    run(
        ["uv", "run", "--no-sync", "maturin", "develop", "--release"],
        cwd=PYTHON_DIR,
        env=env,
    )
    run(["uv", "run", "--no-sync", "pytest", "tests/", "-v"], cwd=PYTHON_DIR, env=env)

    if skip_wheel:
        return

    run(
        ["uv", "run", "--no-sync", "maturin", "build", "--release", "-o", "dist"],
        cwd=PYTHON_DIR,
        env=env,
    )
    wheels = sorted((PYTHON_DIR / "dist").glob("lme_python-*.whl"))
    if not wheels:
        raise CiError("no wheel under python/dist")
    run(
        [
            "uv",
            "run",
            "--no-sync",
            "pip",
            "install",
            "--force-reinstall",
            str(wheels[-1]),
        ],
        cwd=PYTHON_DIR,
        env=env,
    )
    run(["uv", "run", "--no-sync", "pytest", "tests/", "-v"], cwd=PYTHON_DIR, env=env)


def ci(*, reuse_venv: bool = False, skip_wheel: bool = False, skip_python: bool = False) -> None:
    completion_check()
    cargo_build_test()
    if not skip_python:
        python_bindings(reuse_venv=reuse_venv, skip_wheel=skip_wheel)
    lint()
    cargo_check()
    legal_compliance()
    cargo_doctest()
    cargo_doc()
    print(
        "lme_ci.py ci: OK (core jobs; multi-OS matrix, Python 3.10/3.12/3.13, "
        "production-load gates are CI-only)",
        flush=True,
    )


def hooks_install() -> None:
    _require_tool("lefthook")
    run(["lefthook", "install"], cwd=ROOT)
    print("Git hooks installed via lefthook (see lefthook.yml).", flush=True)


def hooks_uninstall() -> None:
    hook = ROOT / ".git" / "hooks" / "pre-commit"
    if hook.is_symlink() or hook.exists():
        hook.unlink(missing_ok=True)
    print("Removed lefthook pre-commit hook if present.", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="lme-rs CI runner (cross-platform; shared by Task, lefthook, GitHub Actions).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fmt", help="Apply rustfmt").set_defaults(fn=lambda _: cargo_fmt(apply=True))
    sub.add_parser("fmt-check", help="Check rustfmt").set_defaults(
        fn=lambda _: cargo_fmt(apply=False)
    )
    sub.add_parser("clippy", help="Run clippy").set_defaults(fn=lambda _: cargo_clippy())
    sub.add_parser("check", help="cargo check --all-targets").set_defaults(
        fn=lambda _: cargo_check()
    )
    sub.add_parser("build-test", help="cargo test (full suite)").set_defaults(
        fn=lambda _: cargo_build_test()
    )
    sub.add_parser("test-fast", help="cargo test --lib only").set_defaults(
        fn=lambda _: cargo_test_fast()
    )
    sub.add_parser("doctest", help="cargo test --doc").set_defaults(fn=lambda _: cargo_doctest())
    sub.add_parser("doc", help="cargo doc").set_defaults(fn=lambda _: cargo_doc())
    sub.add_parser("rust-lint", help="fmt --check + clippy").set_defaults(fn=lambda _: rust_lint())
    sub.add_parser("ruff-lint", help="Ruff check/format on python/tests + examples").set_defaults(
        fn=lambda _: ruff_lint()
    )
    sub.add_parser("lint", help="Rust + Python static checks").set_defaults(fn=lambda _: lint())
    sub.add_parser("audit", help="cargo audit (root + python/) + pip-audit").set_defaults(
        fn=lambda _: audit()
    )
    sub.add_parser("legal", help="validate third-party notices and provenance").set_defaults(
        fn=lambda _: legal_compliance()
    )
    sub.add_parser(
        "preflight",
        help="Pre-push gate: lint + cargo check --all-targets + cargo audit + legal",
    ).set_defaults(fn=lambda _: preflight())
    sub.add_parser(
        "repo-metadata",
        help="Dry-run Cargo.toml metadata sync; verify REPO_ADMIN_TOKEN if set",
    ).set_defaults(fn=lambda _: repo_metadata())
    sub.add_parser(
        "completion-check",
        help="Validate completion-manifest arithmetic and README/report score markers",
    ).set_defaults(fn=lambda _: completion_check())
    sub.add_parser(
        "benchmarks-smoke",
        help="Build release examples + run sleepstudy Rust benchmark once",
    ).set_defaults(fn=lambda _: benchmarks_smoke())
    sub.add_parser(
        "benchmarks-preflight",
        help="benchmarks-smoke + optional R sleepstudy.R when lme4 is installed",
    ).set_defaults(fn=lambda _: benchmarks_preflight())
    sub.add_parser(
        "benchmarks-fair-rust-julia",
        help="Fair fit-only Rust vs Julia timing when Julia + MixedModels are installed (LMM smoke; GLM only for GLMM cases)",
    ).set_defaults(fn=lambda _: benchmarks_fair_rust_julia())
    sub.add_parser(
        "perf-breakdown",
        help="Rust LME_PERF_DIAG vs Julia optsum.feval on crossed_20k fair fixture",
    ).set_defaults(fn=lambda _: perf_breakdown())
    p_bench_fail = sub.add_parser(
        "benchmark-failures",
        help="Print cross-language benchmark failure details from JSON",
    )
    p_bench_fail.add_argument("path")
    p_bench_fail.set_defaults(fn=lambda a: print_benchmark_failures(a.path))
    sub.add_parser("rust-all", help="Rust slice without Python").set_defaults(fn=lambda _: rust_all())

    p_py = sub.add_parser("python", help="Python bindings CI flow")
    p_py.add_argument("--reuse-venv", action="store_true")
    p_py.add_argument("--skip-wheel-reinstall", action="store_true")
    p_py.set_defaults(
        fn=lambda a: python_bindings(
            reuse_venv=a.reuse_venv,
            skip_wheel=a.skip_wheel_reinstall,
        )
    )

    p_ruff = sub.add_parser("ruff-staged", help="Ruff check/format staged python/**/*.py")
    p_ruff.add_argument("--fix", action="store_true")
    p_ruff.set_defaults(fn=lambda a: ruff_staged(fix=a.fix))

    p_r_format = sub.add_parser(
        "r-format-staged",
        help="Format staged comparisons/**/*.R and tests/*.R with styler",
    )
    p_r_format.add_argument("--fix", action="store_true")
    p_r_format.set_defaults(fn=lambda a: r_format_staged(fix=a.fix))

    p_jl_format = sub.add_parser(
        "julia-format-staged",
        help="Format staged comparisons/**/*.jl with JuliaFormatter",
    )
    p_jl_format.add_argument("--fix", action="store_true")
    p_jl_format.set_defaults(fn=lambda a: julia_format_staged(fix=a.fix))

    p_comparison_format = sub.add_parser(
        "comparison-format-check",
        help="Check styler/JuliaFormatter on comparison scripts (skip if tools missing)",
    )
    p_comparison_format.add_argument(
        "--required",
        action="store_true",
        help="Fail when Rscript/styler or julia/JuliaFormatter are unavailable",
    )
    p_comparison_format.set_defaults(
        fn=lambda a: comparison_format_check(required=a.required)
    )

    p_ci = sub.add_parser("ci", help="Full core CI mirror")
    p_ci.add_argument("--reuse-venv", action="store_true")
    p_ci.add_argument("--skip-wheel-reinstall", action="store_true")
    p_ci.add_argument("--skip-python", action="store_true")
    p_ci.set_defaults(
        fn=lambda a: ci(
            reuse_venv=a.reuse_venv,
            skip_wheel=a.skip_wheel_reinstall,
            skip_python=a.skip_python,
        )
    )

    sub.add_parser("hooks-install", help="lefthook install").set_defaults(
        fn=lambda _: hooks_install()
    )
    sub.add_parser("hooks-uninstall", help="Remove lefthook pre-commit hook").set_defaults(
        fn=lambda _: hooks_uninstall()
    )

    args = parser.parse_args(argv)
    os.chdir(ROOT)
    os.environ.setdefault("CARGO_TERM_COLOR", "always")
    try:
        args.fn(args)
    except CiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
