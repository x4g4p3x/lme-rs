"""Guard Windows formatting against Unicode changes under a C locale."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(sys.platform == "win32" and shutil.which("Rscript"), "Windows R required")
class RFormatterTests(unittest.TestCase):
    def test_c_locale_preserves_unicode_strings(self):
        probe = subprocess.run(
            ["Rscript", "-e", 'quit(status=if(requireNamespace("styler", quietly=TRUE)) 0 else 1)'],
            capture_output=True,
        )
        if probe.returncode:
            self.skipTest("R styler required")
        source = 'value <- "Mixed models — θ"\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unicode.R"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run(
                ["Rscript", str(ROOT / "scripts/ci/r_format.R"), str(path)],
                env=dict(os.environ, LC_ALL="C"),
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertEqual(path.read_text(encoding="utf-8"), source)


if __name__ == "__main__":
    unittest.main()
