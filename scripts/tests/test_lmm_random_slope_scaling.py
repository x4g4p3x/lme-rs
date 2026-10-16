"""Scaling-source separation, retained native-fault accounting and cap receipts."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lmm_random_slope_scaling as run  # noqa: E402
import report_lmm_random_slope_scaling as report  # noqa: E402
from lmm_resource_caps import Caps  # noqa: E402
from lmm_stream_store import Journal  # noqa: E402
from verify_lmm_scaling_controls import audit_marker, audit_resume  # noqa: E402


def spec(fault=False):
    return {
        "protocol": run.PROTOCOL,
        "mode": "rehearsal",
        "scenario": "native_fault" if fault else "scaling",
        "namespace": "scaling-v1-unit-native-a" if fault else "scaling-v1-unit-n10-a",
        "cases": run.old.pilot.registry()[:1] if fault else run.old.pilot.registry(),
        "primary_n": 3 if fault else 10,
        "secondary_n": 3 if fault else 10,
        "reference_n": 0 if fault else 10,
        "chunk_size": 8,
        "workers": 1 if fault else 4,
        "variants": run.VARIANTS,
        "caps": Caps().record(),
    }


def timings():
    return dict.fromkeys(
        (
            "start_commit_seconds",
            "outcome_commit_seconds",
            "fit_callback_seconds",
            "input_generation_commit_seconds",
            "validation_recovery_seconds",
        ),
        0.0,
    )


class ScalingTests(unittest.TestCase):
    def test_marker_must_match_owned_pid_and_durable_input(self):
        marker = {
            "environment": "engine",
            "replicate": 1,
            "case": "fixed",
            "input_sha": "rep1",
            "native_pid": 123,
        }
        manifest = {"candidate": "engine", "cases": [{"id": "fixed"}]}
        before = {"inputs": ["rep0", "rep1"], "starts": [(0, "rep0"), (1, "rep1")]}
        receipt = {
            "native_root_pids": [999],
            "resource_guard": {"final_job_process_ids": [999, 123]},
        }
        audit_marker(marker, manifest, before, receipt)
        for key, value in (("native_pid", 456), ("input_sha", "other"), ("case", "other")):
            with self.assertRaises(ValueError):
                audit_marker(dict(marker, **{key: value}), manifest, before, receipt)

    def test_bank_index_rejects_missing_freeze_and_wrong_role(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            manifests = {"n10": spec(), "n20": spec(), "controls/native-fault": spec(True)}
            manifests["n20"].update(
                primary_n=20, secondary_n=20, reference_n=20, namespace="scaling-v1-unit-n20-a"
            )
            for name, manifest in manifests.items():
                manifest["sources"] = {"fixed.py": "unchanged"}
                run.write_new(folder / name / "manifest.json", manifest)
            bank = {
                "manifests": {
                    name: run.old.pilot.shared.digest(folder / name / "manifest.json")
                    for name in manifests
                }
            }
            run.write_new(folder / "bank.json", bank)
            with patch.object(
                run, "verify", side_effect=lambda p: manifests[p.relative_to(folder).as_posix()]
            ):
                run.verify_bank(folder)
                changed = copy.deepcopy(manifests["n20"])
                manifests["n20"] = manifests["n10"]
                with self.assertRaisesRegex(ValueError, "role"):
                    run.verify_bank(folder)
                manifests["n20"] = changed
                (folder / "controls/native-fault/manifest.json").unlink()
                with self.assertRaises(FileNotFoundError):
                    run.verify_bank(folder)

    def test_marker_is_invisible_until_complete_and_cannot_replace_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "native-started.json"
            original = run.write_new

            def saved(pending, value):
                self.assertFalse(path.exists())
                original(pending, value)
                self.assertFalse(path.exists())

            with patch.object(run, "write_new", side_effect=saved):
                run.publish_marker(path, {"native_pid": 123, "input_sha": "complete"})
            self.assertEqual(run.old.pilot.load(path)["input_sha"], "complete")
            with self.assertRaises(FileExistsError):
                run.publish_marker(path, {"input_sha": "replacement"})
            self.assertEqual(run.old.pilot.load(path)["input_sha"], "complete")

    def test_native_fault_audit_rejects_changed_prior_outcome_or_retry(self):
        before = {
            "inputs": ["a", "b", "c"],
            "completed_zero": "saved",
            "starts": [(0, "a", 1.0), (1, "b", 2.0)],
            "outcome_ids": [0],
        }
        after = dict(
            before,
            starts=before["starts"] + [(2, "c", 3.0)],
            outcome_ids=[0, 1, 2],
            interrupted={"fit_error": "interrupted_after_durable_start"},
            last_fit={"converged": True},
        )
        audit_resume(before, after)
        for key, value in (
            ("completed_zero", "replaced"),
            ("starts", [(0, "a", 9.0), (1, "b", 2.0), (2, "c", 3.0)]),
            ("interrupted", {"fit_error": "retried success"}),
            ("outcome_ids", [0, 2]),
        ):
            changed = copy.deepcopy(after)
            changed[key] = value
            with self.assertRaises(ValueError):
                audit_resume(before, changed)

    def test_only_declared_budgets_and_caps_allowed(self):
        for base in (spec(), spec(True)):
            run.validate(base)
            for key, value in (
                ("mode", "confirmation"),
                ("primary_n", 30000),
                ("workers", True),
                ("reference_n", True),
                ("namespace", "scaling-v1-unit-n20-a"),
                ("caps", dict(base["caps"], seconds=1200)),
            ):
                changed = copy.deepcopy(base)
                changed[key] = value
                with self.assertRaises(ValueError):
                    run.validate(changed)

    def test_scaling_and_fault_inputs_are_disjoint(self):
        case = run.old.pilot.registry()[0]
        inputs = [
            run.old.response(case, 0, n)
            for n in ("scaling-v1-n10", "scaling-v1-n20", "scaling-v1-native-fault")
        ]
        self.assertEqual(len({tuple(x) for x in inputs}), 3)
        self.assertNotEqual(inputs[0], run.old.response(case, 0, "20261016-stream-a"))

    def test_interruption_preserves_completed_fit_and_resumes_only_never_started(self):
        with tempfile.TemporaryDirectory() as temp:
            folder, manifest = Path(temp), spec(True)
            calls = []

            def initial(case, row):
                calls.append(row["replicate"])
                if row["replicate"] == 1:
                    raise KeyboardInterrupt()
                return {"converged": False, "methods": {}, "fit_error": "fixed fixture"}

            with self.assertRaises(KeyboardInterrupt):
                run.run_cases(folder, manifest, "candidate", 0, initial, timings())
            path = folder / "cases" / manifest["cases"][0]["id"] / "journal.sqlite"
            journal = Journal(path, run.case_metadata(manifest, manifest["cases"][0]))
            before = journal.db.execute(
                "SELECT payload,sha FROM outcomes WHERE replicate=0"
            ).fetchone()
            start = journal.db.execute("SELECT started FROM starts WHERE replicate=0").fetchone()
            journal.close()
            calls.clear()

            def resumed(case, row):
                calls.append(row["replicate"])
                return {"converged": True, "methods": {}}

            run.run_cases(folder, manifest, "candidate", 0, resumed, timings())
            self.assertEqual(calls, [2])
            journal = Journal(path, run.case_metadata(manifest, manifest["cases"][0]))
            try:
                self.assertEqual(
                    journal.db.execute(
                        "SELECT payload,sha FROM outcomes WHERE replicate=0"
                    ).fetchone(),
                    before,
                )
                self.assertEqual(
                    journal.db.execute("SELECT started FROM starts WHERE replicate=0").fetchone(),
                    start,
                )
                values = list(journal.iter_outcomes("candidate"))
                self.assertEqual(len(values), 3)
                self.assertEqual(values[1][1]["fit_error"], "interrupted_after_durable_start")
            finally:
                journal.close()

    def test_cap_acceptance_rejects_missing_samples_and_wrong_cap_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            folder, manifest = Path(temp), spec()
            run.write_new(folder / "executions/candidate-1-start.json", {"caps": manifest["caps"]})
            run.write_new(
                folder / "executions/candidate-1-end.json",
                {
                    "caps": manifest["caps"],
                    "resource_guard": {"abort_reason": None, "samples": 0},
                    "final_disk": {"bytes": 1},
                },
            )
            self.assertFalse(report.capped_execution_audit(folder, manifest)["accepted"])
            altered = copy.deepcopy(manifest)
            altered["caps"]["seconds"] = 1200
            with self.assertRaises(ValueError):
                report.capped_execution_audit(folder, altered)


if __name__ == "__main__":
    unittest.main()
