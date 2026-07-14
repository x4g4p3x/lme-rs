"""Tests for wheel license packaging (PEP 639 License-File paths)."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci" / "package_wheel_notices.py"
sys.path.insert(0, str(ROOT / "scripts" / "ci"))
from package_wheel_notices import (  # noqa: E402
    _license_file_entry,
    _license_file_wheel_path,
    _metadata_version,
)


def test_metadata_24_license_paths_are_relative_to_licenses_dir() -> None:
    metadata = "Metadata-Version: 2.4\nName: example\n"
    version = _metadata_version(metadata)
    assert version == (2, 4)
    assert _license_file_entry("THIRD_PARTY_NOTICES.md", version) == "THIRD_PARTY_NOTICES.md"
    assert (
        _license_file_wheel_path("pkg-1.0.dist-info/", "THIRD_PARTY_NOTICES.md", version)
        == "pkg-1.0.dist-info/licenses/THIRD_PARTY_NOTICES.md"
    )
    assert (
        _license_file_wheel_path("pkg-1.0.dist-info/", "licenses/THIRD_PARTY_NOTICES.md", version)
        == "pkg-1.0.dist-info/licenses/THIRD_PARTY_NOTICES.md"
    )


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
        meta = next(name for name in names if name.endswith("/METADATA"))
        metadata = archive.read(meta).decode("utf-8")
        assert any(name.endswith("/licenses/THIRD_PARTY_NOTICES.md") for name in names)
        assert not any("/licenses/licenses/" in name for name in names)
        assert "License-File: THIRD_PARTY_NOTICES.md" in metadata
        assert "License-File: licenses/THIRD_PARTY_NOTICES.md" not in metadata

    subprocess.check_call([sys.executable, str(SCRIPT), "--verify-only", str(wheel)])
