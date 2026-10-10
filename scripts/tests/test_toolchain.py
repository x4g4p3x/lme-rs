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

    def minimum_environment(self, directory):
        root = self.environment(
            directory,
            'python-version: ["3.10", "3.11"]\n'
            '  rust-msrv:\n    toolchain: "1.88.0"\n    run: |\n'
            "      cargo +1.88.0 check --locked\n"
            "      cargo +1.88.0 check --locked "
            "--manifest-path python/Cargo.toml --features abi3\n",
        )
        (root / "python").mkdir()
        for manifest in (root / "Cargo.toml", root / "python" / "Cargo.toml"):
            manifest.write_text('rust-version = "1.88"\n')
        (root / "python" / "pyproject.toml").write_text('requires-python = ">=3.10"\n')
        return root

    def test_advertised_minimums_require_real_compatibility_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.minimum_environment(directory)
            with patch.object(ci, "ROOT", root), patch.object(ci, "PYTHON_DIR", root / "python"):
                ci.minimum_versions_check()
                workflow = root / ".github" / "workflows" / "ci.yml"
                workflow.write_text(workflow.read_text().replace('"3.10"', '"3.11"'))
                with self.assertRaisesRegex(ci.CiError, "test Python 3.10"):
                    ci.minimum_versions_check()

    def test_binding_minimum_cannot_drift_from_core(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.minimum_environment(directory)
            (root / "python" / "Cargo.toml").write_text('rust-version = "1.99"\n')
            with patch.object(ci, "ROOT", root), patch.object(ci, "PYTHON_DIR", root / "python"):
                with self.assertRaisesRegex(ci.CiError, "minimum Rust"):
                    ci.minimum_versions_check()

    def test_minimum_job_must_cover_shared_abi_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.minimum_environment(directory)
            workflow = root / ".github" / "workflows" / "ci.yml"
            source = workflow.read_text()
            with patch.object(ci, "ROOT", root), patch.object(ci, "PYTHON_DIR", root / "python"):
                workflow.write_text(source.replace("--features abi3", ""))
                with self.assertRaisesRegex(ci.CiError, "shared ABI"):
                    ci.minimum_versions_check()
                workflow.write_text(source.replace("  rust-msrv:", "  unrelated-job:"))
                with self.assertRaisesRegex(ci.CiError, "dedicated minimum"):
                    ci.minimum_versions_check()

    def test_r_bootstrap_is_excluded_from_comparison_formatting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "comparisons" / "r" / "renv").mkdir(parents=True)
            (root / "comparisons" / "r" / "renv" / "activate.R").write_text("upstream")
            (root / "comparisons" / "sleepstudy.R").write_text("reference")
            with patch.object(ci, "ROOT", root):
                self.assertEqual(
                    ci.comparison_r_files(), ["comparisons/sleepstudy.R", "scripts/ci/restore_r.R"]
                )


if __name__ == "__main__":
    unittest.main()
