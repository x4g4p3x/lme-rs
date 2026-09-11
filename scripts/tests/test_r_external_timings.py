"""R operation timing must retain enough clock resolution to report a duration."""

import json
import shutil
import subprocess
import unittest
from pathlib import Path


class RExternalTimingTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("Rscript"), "Rscript is not installed on PATH")
    def test_fast_inference_has_positive_amortized_samples(self):
        root = Path(__file__).resolve().parents[2]
        packages = subprocess.run(
            [
                "Rscript",
                "-e",
                "quit(status=if(all(vapply(c('lme4','lmerTest','jsonlite'),"
                " requireNamespace, logical(1), quietly=TRUE))) 0L else 1L)",
            ],
            capture_output=True,
            text=True,
        )
        if packages.returncode:
            self.skipTest("R lme4, lmerTest, and jsonlite are required")
        completed = subprocess.run(
            [
                "Rscript",
                "comparisons/bench_external_timings.R",
                "--case",
                "sleepstudy_satterthwaite",
                "--warmups",
                "1",
                "--repeats",
                "2",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        report = json.loads(completed.stdout)
        self.assertNotIn("skipped", report)
        self.assertEqual(len(report["samples_seconds"]), 2)
        self.assertGreaterEqual(report["batch_iterations"], 1)
        self.assertGreater(report["calibration_calls"], 0)
        for sample, elapsed in zip(report["samples_seconds"], report["batch_elapsed_seconds"]):
            self.assertGreater(sample, 0)
            self.assertAlmostEqual(sample * report["batch_iterations"], elapsed)


if __name__ == "__main__":
    unittest.main()
