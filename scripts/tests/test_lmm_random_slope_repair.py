"""New-engine identity, immutable source inventory and held-out seed policy."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lmm_random_slope_repair as run  # noqa: E402


def spec():
    manifest = copy.deepcopy(run.base.old.pilot.load(run.ANCHOR))
    manifest.pop("engine_commit")
    manifest.update(
        protocol=run.PROTOCOL, scenario="repair_pilot", namespace="repair-pilot-v1-unit-a"
    )
    manifest["candidate"]["native_sha256"] = "a" * 64
    manifest["engine_revision"] = {
        "base_commit": "b" * 40,
        "working_tree": True,
        "sources": {n: manifest["sources"][n] for n in manifest["engine_sources"]},
    }
    return manifest


class RepairPilotTests(unittest.TestCase):
    def test_requires_new_native_matching_environment_and_fixed_small_inventory(self):
        manifest = spec()
        run.validate(manifest)
        for key, value in (
            ("mode", "confirmation"),
            ("primary_n", 30000),
            ("workers", True),
            ("namespace", "scaling-v1-old-n20-a"),
            ("reference_n", 0),
        ):
            changed = copy.deepcopy(manifest)
            changed[key] = value
            with self.assertRaises(ValueError):
                run.validate(changed)
        for key, value in (
            ("native_sha256", run.base.old.pilot.load(run.ANCHOR)["candidate"]["native_sha256"]),
            ("polars", "changed"),
        ):
            changed = copy.deepcopy(manifest)
            changed["candidate"][key] = value
            with self.assertRaises(ValueError):
                run.validate(changed)

    def test_build_source_snapshot_must_match_study_identity(self):
        manifest = spec()
        manifest["engine_revision"]["sources"]["src/optimizer.rs"] = "changed"
        with self.assertRaisesRegex(ValueError, "build engine sources"):
            run.validate(manifest)

    def test_fresh_namespace_changes_inputs_and_never_reuses_counterexamples(self):
        case = run.base.old.pilot.registry()[0]
        original = run.base.old.response(case, 0, "scaling-v1-20261016b-n20-a")
        fresh = run.base.old.response(case, 0, "repair-pilot-v1-unit-a")
        self.assertNotEqual(original, fresh)
        self.assertEqual(fresh, run.base.old.response(case, 0, "repair-pilot-v1-unit-a"))

    def test_changed_archived_source_is_rejected(self):
        manifest = spec()
        manifest["sources"] = {**manifest["sources"], **dict.fromkeys(run.ADDITIONS, "a" * 64)}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            build = folder / "engine-build-inputs.json"
            run.write_new(build, manifest["engine_revision"])
            build_hash = run.base.old.pilot.shared.digest(build)
            manifest["build_inputs_sha256"] = build_hash
            run.write_new(folder / "manifest.json", manifest)
            with patch.object(
                run.base.old.pilot.shared,
                "digest",
                side_effect=lambda p: build_hash if Path(p) == build else "changed",
            ):
                with self.assertRaisesRegex(ValueError, "frozen source changed"):
                    run.verify(folder)
            build.write_text("{}")
            with self.assertRaisesRegex(ValueError, "pre-build receipt"):
                run.verify(folder)


if __name__ == "__main__":
    unittest.main()
