#!/usr/bin/env python3
"""Verify the repository's third-party notices, fixture provenance, and Rust license metadata."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_FILES = (
    "THIRD_PARTY_NOTICES.md",
    "RELINKING.md",
    "LICENSES/Apache-2.0.txt",
    "LICENSES/BSD-3-Clause-OpenBLAS.txt",
    "LICENSES/GPL-2.0-or-later.txt",
    "LICENSES/LGPL-2.1-only.txt",
    "LICENSES/Intel-Simplified-Software-License.txt",
    "legal/fixture-provenance.json",
    "legal/dependency-license-exceptions.json",
)


def fail(message: str) -> None:
    raise SystemExit(f"legal compliance check failed: {message}")


def load_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {path.relative_to(ROOT)}: {exc}")


def check_fixture_provenance() -> None:
    payload = load_json(ROOT / "legal/fixture-provenance.json")
    records = payload.get("fixtures")
    if not isinstance(records, list) or not records:
        fail("fixture provenance has no fixture records")

    documented: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            fail("fixture provenance contains a non-object record")
        paths = record.get("paths")
        if not isinstance(paths, list) or not paths:
            fail("fixture provenance record has no paths")
        for raw_path in paths:
            if not isinstance(raw_path, str) or not raw_path.startswith("tests/data/"):
                fail(f"invalid fixture path: {raw_path!r}")
            path = ROOT / raw_path
            if not path.is_file():
                fail(f"documented fixture is missing: {raw_path}")
            documented.add(raw_path)

    actual = {str(path.relative_to(ROOT)).replace("\\", "/") for path in (ROOT / "tests/data").glob("*.csv")}
    missing = sorted(actual - documented)
    if missing:
        fail(f"CSV fixtures missing provenance: {', '.join(missing)}")


def check_cargo_licenses() -> None:
    exceptions = load_json(ROOT / "legal/dependency-license-exceptions.json").get("exceptions", [])
    allowed = {
        (entry["name"], entry["version"])
        for entry in exceptions
        if isinstance(entry, dict) and isinstance(entry.get("name"), str) and isinstance(entry.get("version"), str)
    }
    result = subprocess.run(
        ["cargo", "metadata", "--locked", "--format-version", "1"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    missing = sorted(
        f"{package['name']} {package['version']}"
        for package in metadata["packages"]
        if package.get("source") and not package.get("license") and (package["name"], package["version"]) not in allowed
    )
    if missing:
        fail("unreviewed Rust dependencies without SPDX metadata: " + ", ".join(missing))


def main() -> int:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        fail("missing required legal files: " + ", ".join(missing))
    check_fixture_provenance()
    check_cargo_licenses()
    print("legal compliance check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
