"""Safety and age checks for CI-owned Cargo generations."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "ci" / "ci_artifact_retention.py"
SPEC = importlib.util.spec_from_file_location("ci_artifact_retention", SCRIPT)
retention = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(retention)


class RetentionTests(unittest.TestCase):
    def make_directory_link(self, link: Path, destination: Path) -> None:
        try:
            link.symlink_to(destination, target_is_directory=True)
        except OSError as exc:
            if os.name != "nt":
                self.skipTest(f"creating a directory symlink is unavailable: {exc}")
            env = os.environ.copy()
            env["RETENTION_TEST_LINK"] = str(link)
            env["RETENTION_TEST_DESTINATION"] = str(destination)
            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "New-Item -ItemType Junction -Path $env:RETENTION_TEST_LINK "
                    "-Target $env:RETENTION_TEST_DESTINATION | Out-Null",
                ],
                check=True,
                env=env,
                capture_output=True,
            )

    def test_local_run_uses_owned_target_and_records_completion_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            target = root / "target"
            info = {"workspace_root": str(root), "target_directory": str(target)}
            metadata = mock.Mock(stdout=json.dumps(info))
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch.object(retention.subprocess, "run", return_value=metadata):
                    with mock.patch.object(retention.subprocess, "call", return_value=7) as call:
                        self.assertEqual(retention.run(root, ["example-ci"]), 7)
                        env = call.call_args.kwargs["env"]
            bucket = Path(env["CARGO_TARGET_DIR"])
            self.assertEqual(bucket.parent, root / ".local-ci-generations")
            self.assertFalse(target.exists(), "managed CI must not use the normal Cargo target")
            marker = json.loads((bucket / retention.MARKER).read_text(encoding="utf-8"))
            self.assertTrue(marker["completed"])
            self.assertEqual(marker["root"], str(root))

    def test_cleanup_failure_cannot_hide_the_ci_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            info = {"workspace_root": str(root), "target_directory": str(root / "target")}
            metadata = mock.Mock(stdout=json.dumps(info))
            for ci_result, expected in ((0, 1), (7, 7)):
                with self.subTest(ci_result=ci_result), mock.patch.dict(os.environ, {}, clear=True):
                    with mock.patch.object(retention.subprocess, "run", return_value=metadata):
                        with mock.patch.object(
                            retention.subprocess, "call", return_value=ci_result
                        ):
                            with mock.patch.object(
                                retention, "prune", side_effect=OSError("denied")
                            ):
                                self.assertEqual(retention.run(root, ["example-ci"]), expected)

    def test_ci_can_remove_its_own_target_without_masking_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            info = {"workspace_root": str(root), "target_directory": str(root / "target")}
            metadata = mock.Mock(stdout=json.dumps(info))

            def remove_target(_command, *, cwd, env):
                del cwd
                bucket = Path(env["CARGO_TARGET_DIR"]).resolve(strict=True)
                self.assertEqual(bucket.parent, root / ".local-ci-generations")
                retention.shutil.rmtree(bucket)
                return 0

            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch.object(retention.subprocess, "run", return_value=metadata):
                    with mock.patch.object(retention.subprocess, "call", side_effect=remove_target):
                        self.assertEqual(retention.run(root, ["example-ci"]), 0)

    def test_ci_cannot_replace_its_target_with_a_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            outside = root / "outside"
            outside.mkdir()
            info = {"workspace_root": str(root), "target_directory": str(root / "target")}
            metadata = mock.Mock(stdout=json.dumps(info))
            real_run = subprocess.run
            environment = {
                key: os.environ[key] for key in ("PATH", "SystemRoot") if key in os.environ
            }

            def metadata_or_powershell(command, **kwargs):
                if command[0] == "cargo":
                    return metadata
                return real_run(command, **kwargs)

            def replace_target(_command, *, cwd, env):
                del cwd
                bucket = Path(env["CARGO_TARGET_DIR"]).resolve(strict=True)
                self.assertEqual(bucket.parent, root / ".local-ci-generations")
                retention.shutil.rmtree(bucket)
                self.make_directory_link(bucket, outside)
                return 0

            with mock.patch.dict(os.environ, environment, clear=True):
                with mock.patch.object(
                    retention.subprocess, "run", side_effect=metadata_or_powershell
                ):
                    with mock.patch.object(
                        retention.subprocess, "call", side_effect=replace_target
                    ):
                        self.assertEqual(retention.run(root, ["example-ci"]), 1)
            self.assertFalse((outside / retention.MARKER).exists())

    def test_interrupted_generation_is_not_considered_finished(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            info = {"workspace_root": str(root), "target_directory": str(root / "target")}
            metadata = mock.Mock(stdout=json.dumps(info))
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch.object(retention.subprocess, "run", return_value=metadata):
                    with mock.patch.object(
                        retention.subprocess, "call", side_effect=KeyboardInterrupt
                    ):
                        with self.assertRaises(KeyboardInterrupt):
                            retention.run(root, ["example-ci"])
            bucket = next((root / ".local-ci-generations").glob("generation-*"))
            marker = json.loads((bucket / retention.MARKER).read_text(encoding="utf-8"))
            self.assertFalse(marker["completed"])
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch.object(retention.subprocess, "run", return_value=metadata):
                    with mock.patch.object(retention.subprocess, "call", return_value=0):
                        self.assertEqual(retention.run(root, ["example-ci"]), 0)
            generations = list(bucket.parent.glob("generation-*"))
            self.assertEqual(len(generations), 2)
            self.assertFalse(json.loads((bucket / retention.MARKER).read_text())["completed"])

    def test_malformed_marker_does_not_block_new_ci_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            cache = root / ".local-ci-generations"
            bad = cache / "generation-0-0"
            bad.mkdir(parents=True)
            (bad / retention.MARKER).write_text("[]", encoding="utf-8")
            corrupt = cache / "generation-0-1"
            corrupt.mkdir()
            (corrupt / retention.MARKER).write_text(
                json.dumps({"version": 1, "root": str(root), "completed": True}),
                encoding="utf-8",
            )
            info = {"workspace_root": str(root), "target_directory": str(root / "target")}
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch.object(retention.time, "time", return_value=1):
                    with mock.patch.object(
                        retention.subprocess, "run", return_value=mock.Mock(stdout=json.dumps(info))
                    ):
                        with mock.patch.object(retention.subprocess, "call", return_value=0):
                            self.assertEqual(retention.run(root, ["example-ci"]), 0)
            self.assertTrue(bad.exists())
            self.assertTrue((cache / "generation-0-2" / retention.MARKER).is_file())

    def test_nondefault_cargo_configuration_is_not_redirected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            info = {"workspace_root": str(root), "target_directory": str(root / "shared")}
            metadata = mock.Mock(stdout=json.dumps(info))
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch.object(retention.subprocess, "run", return_value=metadata):
                    with mock.patch.object(retention.subprocess, "call", return_value=0) as call:
                        self.assertEqual(retention.run(root, ["example-ci"]), 0)
                        call.assert_called_once_with(["example-ci"], cwd=root)
            self.assertFalse((root / "target").exists())

    def test_custom_build_directory_and_unavailable_metadata_preserve_ci(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            metadata = mock.Mock(
                stdout=json.dumps(
                    {
                        "workspace_root": str(root),
                        "target_directory": str(root / "target"),
                        "build_directory": str(root / "custom-build"),
                    }
                )
            )
            for outcome in (metadata, subprocess.CalledProcessError(1, "cargo")):
                with self.subTest(outcome=outcome), mock.patch.dict(os.environ, {}, clear=True):
                    with mock.patch.object(retention.subprocess, "run", side_effect=[outcome]):
                        with mock.patch.object(
                            retention.subprocess, "call", return_value=9
                        ) as call:
                            self.assertEqual(retention.run(root, ["example-ci"]), 9)
                            call.assert_called_once_with(["example-ci"], cwd=root)
            self.assertFalse((root / ".local-ci-generations").exists())

    def test_custom_target_in_additional_workspace_skips_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            extra = root / "extra"
            extra.mkdir()
            metadata = [
                mock.Mock(
                    stdout=json.dumps(
                        {"workspace_root": str(root), "target_directory": str(root / "target")}
                    )
                ),
                mock.Mock(
                    stdout=json.dumps(
                        {"workspace_root": str(extra), "target_directory": str(extra / "custom")}
                    )
                ),
            ]
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch.object(retention.subprocess, "run", side_effect=metadata):
                    with mock.patch.object(retention.subprocess, "call", return_value=0) as call:
                        self.assertEqual(retention.run(root, ["example-ci"], (extra,)), 0)
                        call.assert_called_once_with(["example-ci"], cwd=root)
            self.assertFalse((root / "target").exists())

    def test_only_completed_owned_generation_past_boundary_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            cache = root / "cache"
            cache.mkdir()
            now = 2_000_000.0
            cases = {
                "generation-1-0": (True, now - retention.WEEK, str(root)),
                "generation-2-0": (True, now - retention.WEEK + 1, str(root)),
                "generation-3-0": (False, 0, str(root)),
                "generation-4-0": (True, 0, "another-checkout"),
                "generation-5-0": (True, float("nan"), str(root)),
                "generation-6-0": (True, 0, str(root)),
                "generation-7-0": (True, 10**400, str(root)),
            }
            for name, (completed, used, owner) in cases.items():
                bucket = cache / name
                bucket.mkdir()
                (bucket / retention.MARKER).write_text(
                    json.dumps(
                        {"version": 1, "root": owner, "completed": completed, "last_used": used}
                    ),
                    encoding="utf-8",
                )
            boolean_version = cache / "generation-8-0"
            boolean_version.mkdir()
            (boolean_version / retention.MARKER).write_text(
                json.dumps({"version": True, "root": str(root), "completed": True, "last_used": 0}),
                encoding="utf-8",
            )
            current = cache / "generation-6-0"
            with mock.patch.object(retention.shutil, "rmtree") as remove:
                self.assertEqual(
                    retention.prune(cache, root, current, now), [cache / "generation-1-0"]
                )
                remove.assert_called_once_with(cache / "generation-1-0")

    def test_unmarked_directory_and_unrecognised_names_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            cache = root / "cache"
            cache.mkdir()
            (cache / "generation-1-0").mkdir()
            (cache / "user-data").mkdir()
            with mock.patch.object(retention.shutil, "rmtree") as remove:
                self.assertEqual(retention.prune(cache, root, cache / "generation-2-0", 1e9), [])
                remove.assert_not_called()

    def test_cleanup_rejects_a_cache_outside_the_owning_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            outer = Path(tmp).resolve()
            root = outer / "checkout"
            root.mkdir()
            outside = outer / "outside"
            outside.mkdir()
            with mock.patch.object(retention.shutil, "rmtree") as remove:
                with self.assertRaisesRegex(RuntimeError, "outside its owning checkout"):
                    retention.prune(outside, root, outside / "generation-1-0", 1e9)
                remove.assert_not_called()

    def test_prune_removes_owned_generation_and_preserves_other_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            cache = root / "cache"
            cache.mkdir()
            old = cache / "generation-1-0"
            old.mkdir()
            (old / "artifact").write_text("cache", encoding="utf-8")
            retention._write_marker(old, root, completed=True, now=0)
            keep = root / "independent-result"
            keep.write_text("evidence", encoding="utf-8")
            self.assertEqual(retention.prune(cache, root, cache / "generation-2-0", 1e9), [old])
            self.assertFalse(old.exists())
            self.assertEqual(keep.read_text(encoding="utf-8"), "evidence")

    def test_linked_generation_is_not_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            cache = root / "cache"
            cache.mkdir()
            outside = root / "other"
            outside.mkdir()
            (outside / "evidence").write_text("keep", encoding="utf-8")
            retention._write_marker(outside, root, completed=True, now=0)
            self.make_directory_link(cache / "generation-1-0", outside)
            self.assertEqual(retention.prune(cache, root, cache / "generation-2-0", 1e9), [])
            self.assertEqual((outside / "evidence").read_text(encoding="utf-8"), "keep")

    def test_cleanup_does_not_follow_a_link_inside_an_owned_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            cache = root / "cache"
            bucket = cache / "generation-1-0"
            bucket.mkdir(parents=True)
            outside = root / "other"
            outside.mkdir()
            (outside / "evidence").write_text("keep", encoding="utf-8")
            retention._write_marker(bucket, root, completed=True, now=0)
            self.make_directory_link(bucket / "external", outside)
            self.assertEqual(retention.prune(cache, root, cache / "generation-2-0", 1e9), [bucket])
            self.assertFalse(bucket.exists())
            self.assertEqual((outside / "evidence").read_text(encoding="utf-8"), "keep")

    def test_windows_reparse_point_is_treated_as_link(self):
        metadata = types.SimpleNamespace(st_mode=0o040000, st_file_attributes=0x400)
        with mock.patch.object(Path, "lstat", return_value=metadata):
            self.assertTrue(retention._is_link(Path("junction")))

    def test_lock_rejects_overlapping_ci_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "lock"
            with retention._lock(lock), self.assertRaisesRegex(RuntimeError, "another local CI"):
                with retention._lock(lock):
                    pass

    def test_lock_is_exclusive_between_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "lock"
            code = (
                "import importlib.util; "
                f"s=importlib.util.spec_from_file_location('retention',{str(SCRIPT)!r}); "
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                f"ctx=m._lock(__import__('pathlib').Path({str(lock)!r})); ctx.__enter__()"
            )
            with retention._lock(lock):
                proc = subprocess.run(
                    [sys.executable, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("another local CI", proc.stderr)

    def test_explicit_target_and_hosted_ci_keep_original_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for env in (
                {"CI": "true"},
                {"CARGO_TARGET_DIR": "custom"},
                {"CARGO_BUILD_TARGET_DIR": "custom"},
                {"CARGO_BUILD_BUILD_DIR": "custom"},
            ):
                with self.subTest(env=env), mock.patch.dict(os.environ, env, clear=True):
                    with mock.patch.object(retention.subprocess, "call", return_value=13) as call:
                        self.assertEqual(retention.run(root, ["test-command"]), 13)
                        call.assert_called_once_with(["test-command"], cwd=root.resolve())


if __name__ == "__main__":
    unittest.main()
