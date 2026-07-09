#!/usr/bin/env python3
"""Copy required notices into a built wheel and refresh RECORD."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTICE_FILES = (
    ROOT / "THIRD_PARTY_NOTICES.md",
    ROOT / "RELINKING.md",
    ROOT / "LICENSES" / "Apache-2.0.txt",
    ROOT / "LICENSES" / "BSD-3-Clause-OpenBLAS.txt",
    ROOT / "LICENSES" / "GPL-2.0-or-later.txt",
    ROOT / "LICENSES" / "LGPL-2.1-only.txt",
    ROOT / "LICENSES" / "Intel-Simplified-Software-License.txt",
)


def record_hash(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def package_wheel(wheel: Path) -> None:
    if wheel.suffix != ".whl" or not wheel.is_file():
        raise SystemExit(f"not a wheel: {wheel}")
    if any(not path.is_file() for path in NOTICE_FILES):
        raise SystemExit("required legal notice files are missing")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(temp)
        dist_infos = list(temp.glob("*.dist-info"))
        if len(dist_infos) != 1:
            raise SystemExit(f"expected exactly one .dist-info directory in {wheel}")
        dist_info = dist_infos[0]
        licenses = dist_info / "licenses"
        licenses.mkdir(exist_ok=True)
        for source in NOTICE_FILES:
            shutil.copy2(source, licenses / source.name)

        metadata = dist_info / "METADATA"
        metadata_text = metadata.read_text(encoding="utf-8").rstrip()
        existing_headers = set(metadata_text.splitlines())
        new_headers = [
            f"License-File: licenses/{path.name}"
            for path in NOTICE_FILES
            if f"License-File: licenses/{path.name}" not in existing_headers
        ]
        if new_headers:
            metadata.write_text(
                metadata_text + "\n" + "\n".join(new_headers) + "\n",
                encoding="utf-8",
            )
        record = dist_info / "RECORD"
        with record.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            for path in sorted(temp.rglob("*")):
                if path.is_file() and path != record:
                    writer.writerow((str(path.relative_to(temp)).replace("\\", "/"), record_hash(path), path.stat().st_size))
            writer.writerow((str(record.relative_to(temp)).replace("\\", "/"), "", ""))
        with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(temp.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(temp))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    package_wheel(args.wheel)
    print(f"added third-party notices to {args.wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
