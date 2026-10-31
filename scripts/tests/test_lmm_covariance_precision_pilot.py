"""Prospective controls, new-engine identity and immutable evidence."""

import copy
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lmm_covariance_precision_pilot as run  # noqa: E402


def spec():
    manifest = copy.deepcopy(run.base.old.pilot.load(run.ANCHOR))
    manifest.update(
        caps_controls={"result.json": "a" * 64},
        protocol=run.PROTOCOL,
        scenario="covariance_precision_pilot",
        namespace="covariance-precision-v1-unit-a",
        control=dict(run.CONTROL),
        reuse_policy="new-source-identified-covariance-repair-v1",
        predecessor_manifest_sha256=run.base.old.pilot.shared.digest(run.ANCHOR),
        historical_runs=[
            {
                "namespace": manifest["namespace"],
                "sha256": run.base.old.pilot.shared.digest(run.ANCHOR),
            }
        ],
        study_revision={"base_commit": "b" * 40, "working_tree": True},
    )
    manifest["candidate"]["native_sha256"] = "a" * 64
    manifest["wheel_sha256"] = "b" * 64
    manifest["engine_revision"]["sources"] = run.engine_inputs()
    manifest["engine_revision"]["base_commit"] = "b" * 40
    manifest["sources"].update(manifest["engine_revision"]["sources"])
    return manifest


class CovariancePrecisionPilotTests(unittest.TestCase):
    def test_exact_bits_match_the_original_immutable_counterexamples(self):
        source = run.ROOT / "scientific/STOPPING_CONTROL_PILOT_V1_COUNTEREXAMPLES.json"
        retained = json.loads(source.read_bytes())["counterexamples"]
        derivative = json.loads(
            (run.ROOT / "tests/data/stopping_control_counterexample_bits.json").read_bytes()
        )
        self.assertEqual(derivative["source_sha256"], run.base.old.pilot.shared.digest(source))
        self.assertEqual(len(derivative["fixtures"]), len(retained))
        for original, bits in zip(retained, derivative["fixtures"]):
            self.assertEqual(bits["case"], original["case"]["id"])
            self.assertEqual(bits["replicate"], original["replicate"])
            self.assertEqual(bits["oracle"], original["candidate_numerical"]["oracle"])
            for name, values in (
                ("x", original["case"]["x"]),
                ("y", original["y"]),
                ("theta", original["candidate"]["theta"]),
            ):
                expected = [struct.unpack("<Q", struct.pack("<d", v))[0] for v in values]
                self.assertEqual(bits[name + "_bits"], expected)

    def test_cap_controls_reject_success_and_unsupported_memory_claim(self):
        import verify_lmm_covariance_caps as controls

        for mode in ("success", "memory", "substitution"):

            def fake(args, caps, folder, receipts, label):
                receipts.mkdir(parents=True)
                identity = {
                    "caps": caps.record(),
                    "label": label,
                    "command": controls.command_identity(args),
                    "inputs": {},
                    "thread_limits": controls.THREAD_LIMITS,
                }
                if mode == "substitution" and label == "time":
                    identity["command"]["argv_sha256"] = "a" * 64
                    plan_path = folder.parent / "plan.json"
                    plan = json.loads(plan_path.read_bytes())
                    plan["cases"][0]["command"] = identity["command"]
                    plan_path.write_text(json.dumps(plan))
                start = receipts / (label + "-1-start.json")
                controls.write_new(start, identity)
                reason = {
                    "time": "phase_time_cap",
                    "disk": "sampled_disk_cap",
                    "memory": "os_job_charge_above_limit",
                }[label]
                end = {
                    "identity": identity,
                    "caps": caps.record(),
                    "outputs": {},
                    "exit_codes": [
                        0
                        if mode == "success"
                        else 91
                        if label == "memory" and mode == "substitution"
                        else -1
                    ],
                    "start_sha256": controls.digest(start),
                    "resource_guard": {
                        "samples": 1,
                        "final_active_job_processes": 0,
                        "abort_reason": reason,
                        "job_events": [],
                    },
                }
                name = label + "-1-end.json"
                controls.write_new(receipts / name, end)
                (receipts / (label + "-1.log")).write_text(
                    "allocation_denied" if mode == "substitution" and label == "memory" else ""
                )
                return {**end, "receipt": name}

            with (
                tempfile.TemporaryDirectory() as directory,
                patch.object(controls, "run_capped", side_effect=fake),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "raw receipt"
                    if mode == "success"
                    else "hard memory"
                    if mode == "memory"
                    else "command invocation",
                ):
                    controls.controls(Path(directory) / "controls", Path(sys.executable))

    def test_new_engine_requires_compatible_environment_and_new_native(self):
        manifest = spec()
        run.validate(manifest)
        anchor = run.base.old.pilot.load(run.ANCHOR)
        for key, value in (
            ("native_sha256", anchor["candidate"]["native_sha256"]),
            ("python", "changed"),
            ("polars", "changed"),
            ("platform", "changed"),
            ("version", "changed"),
        ):
            changed = copy.deepcopy(manifest)
            changed["candidate"][key] = value
            with self.assertRaisesRegex(ValueError, "new repaired engine"):
                run.validate(changed)
        for key, value in (
            ("wheel_sha256", anchor["wheel_sha256"]),
            ("reuse_policy", "old-engine"),
        ):
            changed = copy.deepcopy(manifest)
            changed[key] = value
            with self.assertRaises(ValueError):
                run.validate(changed)

    def test_changed_control_inventory_and_seed_policy_rejected(self):
        manifest = spec()
        for key, value in (
            ("control", {**run.CONTROL, "tolerance": 1e-12}),
            ("control", {**run.CONTROL, "max_iterations": 10000}),
            ("mode", "confirmation"),
            ("primary_n", 30000),
            ("workers", True),
            ("reference_n", 0),
            ("variants", []),
            ("namespace", "repair-pilot-v1-used-a"),
            ("namespace", "covariance-precision-v1-" + "a" * 44),
            ("historical_runs", []),
        ):
            changed = copy.deepcopy(manifest)
            changed[key] = value
            with self.assertRaises(ValueError):
                run.validate(changed)
        manifest["historical_runs"].append({"namespace": manifest["namespace"], "sha256": "c" * 64})
        with self.assertRaisesRegex(ValueError, "already used"):
            run.validate(manifest)

    def test_build_and_study_revisions_are_distinct_and_cannot_mislabel_engine(self):
        manifest = spec()
        self.assertNotEqual(manifest["study_revision"], manifest["engine_revision"])
        for change, value in (("base_commit", "invalid"), ("sources", {})):
            changed = copy.deepcopy(manifest)
            changed["engine_revision"][change] = value
            with self.assertRaises(ValueError):
                run.validate(changed)
        manifest["study_revision"]["working_tree"] = "false"
        with self.assertRaisesRegex(ValueError, "actual study"):
            run.validate(manifest)

    def test_explicit_candidate_control_does_not_mutate_historical_helper(self):
        historical = dict(run.base.old.pilot.CONTROL)
        fit = SimpleNamespace(
            coefficients=[1.0, 0.5],
            theta=[1.0, 0.25, 1.0],
            sigma2=1.0,
            reml_criterion=10.0,
            converged=False,
            diagnostics={"termination": "IterationLimit"},
        )
        lme = SimpleNamespace(
            lmer=Mock(return_value=fit), FitControl=Mock(side_effect=lambda **kw: kw)
        )
        result = run.candidate_fit(lme, object())
        self.assertEqual(lme.FitControl.call_args.kwargs, run.CONTROL)
        self.assertEqual(lme.lmer.call_args.kwargs["control"], run.CONTROL)
        self.assertEqual(result["fit_error"], "nonconvergence")
        self.assertEqual(result["methods"], {})
        self.assertEqual(run.base.old.pilot.CONTROL, historical)

    def test_archive_and_build_receipt_tampering_rejected(self):
        manifest = spec()
        manifest["sources"] = {**manifest["sources"], **dict.fromkeys(run.ADDITIONS, "a" * 64)}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            build = folder / "engine-build-inputs.json"
            run.write_new(build, manifest["engine_revision"])
            build_hash = run.base.old.pilot.shared.digest(build)
            manifest["build_inputs_sha256"] = build_hash
            run.write_new(folder / "manifest.json", manifest)
            anchor_hash = run.base.old.pilot.shared.digest(run.ANCHOR)
            with patch.object(
                run.base.old.pilot.shared,
                "digest",
                side_effect=lambda p: (
                    build_hash
                    if Path(p) == build
                    else anchor_hash
                    if Path(p) == run.ANCHOR
                    else "changed"
                ),
            ):
                with self.assertRaisesRegex(ValueError, "frozen source changed"):
                    run.verify(folder)
            build.write_text("{}")
            with self.assertRaisesRegex(ValueError, "pre-build receipt"):
                run.verify(folder)

    def test_method_exception_remains_failed_and_does_not_escape_readiness(self):
        fit = SimpleNamespace(
            coefficients=[1.0, 0.5],
            theta=[1.0, 0.25, 1.0],
            sigma2=1.0,
            reml_criterion=10.0,
            converged=True,
            diagnostics={},
        )
        lme = SimpleNamespace(lmer=Mock(return_value=fit), FitControl=Mock())
        with patch.object(
            run.base.old.pilot.shared,
            "evaluate_candidate_methods",
            side_effect=ValueError("injected"),
        ):
            result = run.candidate_fit(lme, object())
        self.assertFalse(result["converged"])
        self.assertEqual(result["methods"], {})
        self.assertIn("injected", result["fit_error"])

    def test_corrupted_saved_input_is_rejected_before_callback(self):
        from lmm_stream_store import Journal

        manifest = spec()
        manifest["cases"] = manifest["cases"][:1]
        case = manifest["cases"][0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "cases" / case["id"]
            path.mkdir(parents=True)
            journal = Journal(path / "journal.sqlite", run.case_metadata(manifest, case))
            journal.add_chunk([{"case": case["id"], "replicate": 0, "y": [0.0] * len(case["x"])}])
            journal.db.execute("DROP TRIGGER immutable_chunks_UPDATE")
            journal.db.execute("UPDATE chunks SET payload=?", (b"corrupted",))
            journal.db.commit()
            journal.close()
            callback = Mock()
            timings = dict.fromkeys(
                (
                    "validation_recovery_seconds",
                    "input_generation_commit_seconds",
                    "start_commit_seconds",
                    "outcome_commit_seconds",
                    "fit_callback_seconds",
                ),
                0.0,
            )
            with self.assertRaisesRegex(ValueError, "checksum"):
                run.base.run_cases(root, manifest, "candidate", 0, callback, timings)
            callback.assert_not_called()

    def test_overlap_and_overwrite_rejected_before_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for target, anchor in ((root, root), (root / "new", root), (root, root / "old")):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    run.freeze(target, anchor, None, None, None, "unused", None)
            target = root / "existing"
            target.mkdir()
            with self.assertRaises(FileExistsError):
                run.freeze(target, root / "old", None, None, None, "unused", None)

    def test_test_namespace_is_deterministic_and_disjoint_from_old_inputs(self):
        case = run.base.old.pilot.registry()[0]
        fresh = run.base.old.response(case, 0, "covariance-precision-v1-unit-a")
        old = run.base.old.response(case, 0, "repair-pilot-v1-unit-a")
        self.assertNotEqual(fresh, old)
        self.assertEqual(fresh, run.base.old.response(case, 0, "covariance-precision-v1-unit-a"))


if __name__ == "__main__":
    unittest.main()
