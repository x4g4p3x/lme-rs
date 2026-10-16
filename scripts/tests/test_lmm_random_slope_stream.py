"""Execution order, no-retry accounting and streamed analysis regressions."""

import copy
import gzip
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lmm_random_slope_stream as run  # noqa: E402
import lmm_random_slope_v2 as pilot  # noqa: E402
import plan_lmm_random_slope_confirmation as planning  # noqa: E402
import report_lmm_random_slope_stream as report  # noqa: E402
from lmm_stream_store import Journal, canonical, sha  # noqa: E402


def manifest(cases=None, chunk=2):
    return {
        "mode": "rehearsal",
        "protocol": run.PROTOCOL,
        "namespace": "unit-test",
        "cases": pilot.registry() if cases is None else cases,
        "variants": run.VARIANTS,
        "primary_n": 2,
        "secondary_n": 2,
        "reference_n": 1,
        "chunk_size": chunk,
    }


def fit(criterion=100):
    return {
        "converged": True,
        "beta": 0.1,
        "criterion": criterion,
        "sigma2": 1.0,
        "covariance": [[0.5, 0.0], [0.0, 0.25]],
        "methods": {"normal": {"beta": 0.1, "se": 0.1, "f": 1.0, "p": math.erfc(1 / math.sqrt(2))}},
    }


class ExecutionTests(unittest.TestCase):
    def test_process_group_reaps_child_after_root_crashes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "child.pid"
            code = """import subprocess,sys,os
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])
Path(sys.argv[1]).write_text(str(child.pid))
os._exit(17)
"""
            group = run.ProcessGroup()
            try:
                root = group.start([sys.executable, "-c", code, str(path)])
                self.assertEqual(root.wait(timeout=10), 17)
                child_pid = int(path.read_text())
                if os.name == "nt":
                    self.assertIsNotNone(run.peak_working_set(child_pid))
            finally:
                group.close()
            for _ in range(100):
                if os.name == "nt":
                    dead = run.peak_working_set(child_pid) is None
                else:
                    status = Path(f"/proc/{child_pid}/stat")
                    dead = not status.exists() or status.read_text().split()[2] == "Z"
                if dead:
                    break
                time.sleep(0.01)
            self.assertTrue(dead, "orphan native child survived owned process group")

    def test_failed_r_startup_closes_owned_processes(self):
        with (
            patch.object(run.RSession, "receive", side_effect=RuntimeError("bad startup")),
            patch.object(run.ProcessGroup, "close", autospec=True) as close,
        ):
            with tempfile.TemporaryFile(mode="w+") as log:
                process = unittest.mock.MagicMock()
                process.stdout = io.StringIO("")
                with patch.object(run.ProcessGroup, "start", return_value=process):
                    with self.assertRaisesRegex(RuntimeError, "bad startup"):
                        run.RSession(Path("unused"), log)
            self.assertTrue(close.called)

    def test_complete_fits_cannot_hide_failed_worker_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            spec = manifest()
            run.write_new(
                folder / "executions/candidate-1-start.json",
                {"phase": "candidate", "workers": 1, "manifest_sha": sha(canonical(spec))},
            )
            run.write_new(folder / "executions/candidate-1-end.json", {"exit_codes": [1]})
            audited = report.execution_audit(folder, spec)
            self.assertFalse(audited["accepted"])
            self.assertTrue(any("worker exits" in x for x in audited["failures"]))
            self.assertTrue(any("missing worker" in x for x in audited["failures"]))
            lock = pilot.load(run.ROOT / "scientific/renv.lock")
            run.write_new(
                folder / "executions/reference-2-start.json",
                {"phase": "reference", "workers": 1, "manifest_sha": sha(canonical(spec))},
            )
            run.write_new(folder / "executions/reference-2-end.json", {"exit_codes": [0]})
            run.write_new(
                folder / "executions/reference-2-worker-0.json",
                {
                    "phase": "reference",
                    "worker": 0,
                    "manifest_sha": sha(canonical(spec)),
                    "native_python_pid": 1,
                    "python_peak_working_set_bytes": 1024,
                    "r_peak_working_set_bytes": 1024,
                    "r_environment": {
                        "native_pid": 2,
                        "r_version": "R version " + lock["R"]["Version"] + " (fixture)",
                        "namespaces": dict(lock["LoadedNamespaces"], Unknown=lock["R"]["Version"]),
                    },
                },
            )
            with self.assertRaisesRegex(ValueError, "namespace version differs"):
                report.execution_audit(folder, spec)

    def test_rehearsal_budget_and_family_cannot_be_expanded(self):
        base = manifest()
        run.validate_design(base)
        for key, value in (
            ("mode", "confirmation"),
            ("primary_n", 30000),
            ("secondary_n", True),
            ("reference_n", 3),
            ("cases", base["cases"][:-1]),
        ):
            bad = dict(base, **{key: value})
            with self.assertRaises(ValueError):
                run.validate_design(bad)

    def test_generator_is_disjoint_and_independent_of_order(self):
        for case in pilot.smoke_cases():
            expected = [run.response(case, i, "test-a") for i in range(3)]
            self.assertEqual(
                expected,
                list(reversed([run.response(case, i, "test-a") for i in reversed(range(3))])),
            )
            self.assertNotEqual(expected[0], pilot.response(case, 0, "pilot"))
            self.assertNotEqual(expected[0], pilot.response(case, 0, "smoke"))
            self.assertNotEqual(expected[0], run.response(case, 0, "test-b"))

    def test_changed_chunks_and_worker_order_do_not_change_inputs_or_repeat_fits(self):
        cases = pilot.smoke_cases()
        observed = []
        with tempfile.TemporaryDirectory() as temp:
            for version, (chunk, workers, order) in enumerate(((1, 1, [0]), (4, 2, [1, 0]))):
                folder, spec = Path(temp) / str(version), manifest(cases, chunk)
                calls = []

                def callback(c, r):
                    calls.append((c["id"], r["replicate"]))
                    return {"converged": False, "methods": {}, "fit_error": "controlled fixture"}

                for index in order:
                    run.run_cases(folder, spec, "candidate", index, workers, callback)
                self.assertEqual(len(calls), len(cases) * 2)
                calls.clear()
                run.run_cases(folder, spec, "candidate", 0, 1, callback)
                self.assertEqual(calls, [])
                inputs = []
                for case in cases:
                    journal = Journal(
                        folder / "cases" / case["id"] / "journal.sqlite",
                        run.case_metadata(spec, case),
                    )
                    try:
                        inputs.extend(sha(canonical(row)) for row in journal.iter_inputs())
                    finally:
                        journal.close()
                observed.append(inputs)
        self.assertEqual(*observed)

    def test_failed_and_interrupted_fits_remain_counted_without_callback_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            case = pilot.registry()[0]
            spec = manifest([case])
            journal = Journal(Path(temp) / "case.sqlite", run.case_metadata(spec, case))
            try:
                rows = [
                    {"case": case["id"], "replicate": i, "y": run.response(case, i, "unit-test")}
                    for i in range(2)
                ]
                journal.add_chunk(rows)
                with self.assertRaises(KeyboardInterrupt):
                    run.execute(
                        journal,
                        "candidate",
                        rows[0],
                        lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
                    )
                journal.recover()
                run.execute(
                    journal, "candidate", rows[0], lambda: self.fail("interrupted fit retried")
                )
                run.execute(
                    journal,
                    "candidate",
                    rows[1],
                    lambda: (_ for _ in ()).throw(ValueError("bad fit")),
                )
                values = list(journal.iter_outcomes("candidate"))
                self.assertEqual(len(values), 2)
                self.assertIn("interrupted", values[0][1]["fit_error"])
                self.assertEqual(values[1][1]["fit_error"], "ValueError: bad fit")
            finally:
                journal.close()

    def test_generic_reference_selection_ignores_inference_and_oracle(self):
        a, b, c = fit(100), fit(99), fit(98)
        b["methods"] = {"satterthwaite": {"error": "inference failed"}}
        c["converged"] = False
        self.assertEqual(report.choose_reference([a, b, c]), 1)
        self.assertIsNone(report.choose_reference([c]))
        self.assertEqual(report.choose_reference([a, copy.deepcopy(a)]), 0)
        case = next(c for c in pilot.registry() if c["stratum"] == "imbalanced")
        self.assertEqual(
            report.numerical(case, run.response(case, 0, "unit-test"), b), (None, True)
        )
        self.assertIsNone(report.decision(b, "satterthwaite", 0.0, True))
        for field in ("beta", "criterion", "sigma2"):
            self.assertFalse(report.fit_eligible(dict(a, **{field: True})))

    @unittest.skipIf(planning.scipy is None, "optional SciPy required for inference analysis")
    def test_invalid_method_payloads_cannot_turn_into_valid_nonrejections(self):
        base = fit()
        self.assertIsNotNone(report.decision(base, "normal", 0.0, True))
        for field, value in (
            ("p", 0.001),
            ("p", float("nan")),
            ("p", True),
            ("se", 0),
            ("f", 2),
            ("beta", 0.2),
        ):
            bad = copy.deepcopy(base)
            bad["methods"]["normal"][field] = value
            self.assertIsNone(report.decision(bad, "normal", 0.0, True))
        self.assertIsNone(report.decision(dict(base, methods=None), "normal", 0.0, True))
        self.assertIsNone(report.decision(base, "normal", 0.0, False))

    def test_native_memory_counter_is_for_actual_process(self):
        if sys.platform not in ("win32", "linux"):
            self.skipTest("native OS memory counter not supported")
        self.assertGreater(run.peak_working_set(), 1024 * 1024)

    @unittest.skipIf(planning.scipy is None, "optional SciPy required for report")
    def test_full_streamed_failure_counts_keep_all_72_claims(self):
        with tempfile.TemporaryDirectory() as temp:
            folder, output = Path(temp) / "run", Path(temp) / "report"
            spec = manifest()
            folder.mkdir()
            run.write_new(folder / "manifest.json", spec)

            def failure(*_):
                return {"converged": False, "methods": {}, "fit_error": "fixture"}

            run.run_cases(folder, spec, "candidate", 0, 1, failure)
            run.run_cases(folder, spec, "reference", 0, 1, failure)
            with patch.object(run, "verify", return_value=spec):
                report.report(folder, output)
            summary = json.loads((output / "summary.json").read_text())
            self.assertFalse(summary["confirmation_claim"])
            self.assertFalse(summary["execution_acceptance"]["accepted"])
            for count in summary["reference_variant_attempts"].values():
                self.assertEqual((count["attempted"], count["fit_error"]), (94, 94))
            self.assertEqual(len(summary["primary"]), 72)
            self.assertEqual(len(summary["counts"]), 94 * 2 * 3)
            for row in summary["primary"]:
                self.assertEqual(
                    (row["n"], row["k"], row["m"], row["bounds"], row["rule_pass"]),
                    (2, 0, 2, [0.0, 1.0], False),
                )
            self.assertEqual(
                sum(r["attempted"] for r in summary["counts"] if r["engine"] == "candidate"),
                188 * 3,
            )
            rscript = shutil.which("Rscript")
            if rscript:
                checker = run.ROOT / "comparisons/scientific/lmm_random_slope_stream_check.R"
                result = subprocess.run(
                    [
                        rscript,
                        str(checker),
                        str(folder / "manifest.json"),
                        str(output),
                        str(output / "r-check.json"),
                    ],
                    cwd=run.ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                original_audit = (output / "audit.jsonl.gz").read_bytes()
                for index, kind in enumerate(
                    ("primary", "counts", "audit_sha256", "bounds", "oracle")
                ):
                    altered = copy.deepcopy(summary)
                    if kind == "oracle":
                        lines = gzip.decompress(original_audit).splitlines()
                        raw = json.loads(lines[0])
                        raw["candidate_numerical"] = None
                        lines[0] = canonical(raw)
                        changed = gzip.compress(b"\n".join(lines) + b"\n", mtime=0)
                        (output / "audit.jsonl.gz").write_bytes(changed)
                        altered["audit_sha256"] = sha(changed)
                    elif kind == "bounds":
                        del altered["primary"][0]["bounds"]
                    elif kind == "audit_sha256":
                        altered[kind] = "0" * 64
                    else:
                        altered[kind] = [altered[kind][0]] * len(altered[kind])
                    # Controlled corruption in a disposable fixture, never a study record.
                    (output / "summary.json").write_text(json.dumps(altered))
                    result = subprocess.run(
                        [
                            rscript,
                            str(checker),
                            str(folder / "manifest.json"),
                            str(output),
                            str(output / f"invalid-{index}.json"),
                        ],
                        cwd=run.ROOT,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
