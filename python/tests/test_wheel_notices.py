"""Tests for wheel license packaging (PEP 639 License-File paths)."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci" / "package_wheel_notices.py"


def _find_built_wheel() -> Path | None:
    dist = Path(__file__).resolve().parents[1] / "dist"
    wheels = sorted(dist.glob("lme_python-*.whl"))
    return wheels[-1] if wheels else None


def test_package_wheel_repairs_nested_license_paths(tmp_path: Path) -> None:
    source = _find_built_wheel()
    if source is None:
        pytest.skip("no built wheel in python/dist; run maturin build first")
    wheel = tmp_path / source.name
    wheel.write_bytes(source.read_bytes())

    with zipfile.ZipFile(wheel) as archive:
        meta = next(name for name in archive.namelist() if name.endswith("/METADATA"))
        prefix = meta[: -len("METADATA")]
    nested = f"{prefix}licenses/licenses/THIRD_PARTY_NOTICES.md"

    with zipfile.ZipFile(wheel, "a") as archive:
        if nested not in archive.namelist():
            archive.writestr(nested, b"stale nested copy")

    subprocess.check_call([sys.executable, str(SCRIPT), str(wheel)])

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert not any("/licenses/licenses/" in name for name in names)
        assert any(name.endswith("/licenses/THIRD_PARTY_NOTICES.md") for name in names)


def test_package_wheel_notices_idempotent_and_verifiable(tmp_path: Path) -> None:
    source = _find_built_wheel()
    if source is None:
        pytest.skip("no built wheel in python/dist; run maturin build first")
    wheel = tmp_path / source.name
    wheel.write_bytes(source.read_bytes())

    for _ in range(2):
        subprocess.check_call([sys.executable, str(SCRIPT), str(wheel)])

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert any(name.endswith("/licenses/THIRD_PARTY_NOTICES.md") for name in names)
        assert not any("/licenses/licenses/" in name for name in names)

    subprocess.check_call([sys.executable, str(SCRIPT), "--verify-only", str(wheel)])
