"""Check that accidental tool-version drift cannot bypass preflight."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "lme_ci", Path(__file__).resolve().parents[1] / "ci" / "lme_ci.py"
)
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)


class ToolchainTests(unittest.TestCase):
    def environment(self, directory, workflow):
        root = Path(directory)
        (root / ".github" / "workflows").mkdir(parents=True)
        (root / "mise.toml").write_text('rust = "1.99.0"\nuv = "0.11.24"\n')
        (root / "rust-toolchain.toml").write_text('channel = "1.99.0"\n')
        (root / ".github" / "workflows" / "ci.yml").write_text(workflow)
        return root

    def test_installer_action_pin_does_not_replace_uv_version_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.environment(
                directory,
                "      - uses: astral-sh/setup-uv@abcdef\n        with:\n"
                '          python-version: "3.11"\n',
            )
            with patch.object(ci, "ROOT", root):
                with self.assertRaisesRegex(ci.CiError, "uv version"):
                    ci.toolchain_check()

    def test_main_validation_rejects_moving_rust_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.environment(directory, "          toolchain: stable\n")
            with patch.object(ci, "ROOT", root):
                with self.assertRaisesRegex(ci.CiError, "Rust version"):
                    ci.toolchain_check()

    def test_separate_latest_compiler_job_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.environment(
                directory,
                '          toolchain: "1.99.0"\n  rust-latest:\n          toolchain: stable\n',
            )
            with patch.object(ci, "ROOT", root):
                ci.toolchain_check()

    def test_local_compiler_pins_must_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.environment(directory, "")
            (root / "rust-toolchain.toml").write_text('channel = "1.98.0"\n')
            with patch.object(ci, "ROOT", root):
                with self.assertRaisesRegex(ci.CiError, "versions differ"):
                    ci.toolchain_check()

    def test_moving_local_versions_are_not_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.environment(directory, "")
            (root / "mise.toml").write_text('rust = "stable"\nuv = "latest"\n')
            with patch.object(ci, "ROOT", root):
                with self.assertRaisesRegex(ci.CiError, "exact rust version"):
                    ci.toolchain_check()


if __name__ == "__main__":
    unittest.main()
