"""Ensure alternate build directories and optimizer labels are honored."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_cross_language_benchmarks as cross
import run_external_timings as external
import run_perf_breakdown as perf


class BenchmarkRuntimeTests(unittest.TestCase):
    def test_phase_backend_mismatch_is_a_failed_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            args = SimpleNamespace(
                cases="random_intercept_10k",
                implementations="rust",
                warmups=0,
                data_dir=directory,
                julia=None,
                skip_build=True,
                rust_features="",
                output=str(output),
            )
            with patch.object(perf, "parse_args", return_value=args), patch.object(
                perf, "resolve_julia", return_value=None
            ), patch.object(
                perf, "ensure_data", return_value=Path(directory) / "data.csv"
            ), patch.object(perf, "rust_breakdown", return_value={"optimizer_backend": "basin"}):
                self.assertEqual(perf.main(), 1)
            report = json.loads(output.read_text(encoding="utf-8"))[0]
            self.assertIsNone(report["rust"])
            self.assertIn("optimizer_backend=argmin", report["failures"][0]["error"])

    def test_external_rejects_a_wrong_or_missing_backend(self):
        for actual in [None, "argmin"]:
            completed = subprocess.CompletedProcess(
                [], 0, json.dumps({"optimizer_backend": actual})
            )
            with patch.object(external, "_run", return_value=completed):
                with self.assertRaisesRegex(ValueError, "optimizer_backend=basin"):
                    external.rust_report("orange_nlmer", 2, 10, "basin")

    def test_external_passes_the_feature_to_cargo(self):
        completed = subprocess.CompletedProcess([], 0, '{"optimizer_backend": "basin"}')
        with patch.object(external, "_run", return_value=completed) as run:
            external.rust_report("orange_nlmer", 2, 10, "basin")
        command = run.call_args.args[0]
        self.assertLess(command.index("--features"), command.index("--"))
        self.assertEqual(command[command.index("--features") + 1], "basin")

    def test_cross_language_resolves_relative_and_absolute_cargo_targets(self):
        suffix = ".exe" if os.name == "nt" else ""
        for target in ["alternate-target", str(cross.REPO_ROOT / "absolute-target")]:
            with patch.dict(os.environ, {"CARGO_TARGET_DIR": target}):
                expected = Path(target)
                if not expected.is_absolute():
                    expected = cross.REPO_ROOT / expected
                self.assertEqual(
                    cross.rust_binary_path("sleepstudy"),
                    expected / "release" / "examples" / f"sleepstudy{suffix}",
                )
                self.assertEqual(
                    perf.rust_binary("sleepstudy"),
                    expected / "release" / "examples" / f"sleepstudy{suffix}",
                )


if __name__ == "__main__":
    unittest.main()
