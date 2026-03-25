"""Resolve paths to the repository root and `tests/data` for bundled fixtures."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    # python/examples/verification_project/paths.py -> parents[3] == repo root
    return Path(__file__).resolve().parents[3]


def tests_data(*parts: str) -> Path:
    return repo_root().joinpath("tests", "data", *parts)
