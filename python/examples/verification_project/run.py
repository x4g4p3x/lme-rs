#!/usr/bin/env python3
"""
Run all parity checks (same assertions as `test_parity.py`).

From the repository root, after building the extension:

    cd python && maturin develop --release
    python examples/verification_project/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parity import run_all_checks  # noqa: E402


def main() -> int:
    results = run_all_checks()
    failed = 0
    for name, err in results:
        if err is None:
            print(f"  OK   {name}")
        else:
            failed += 1
            print(f"  FAIL {name}: {err}")
    print()
    if failed:
        print(f"{failed} check(s) failed.")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
