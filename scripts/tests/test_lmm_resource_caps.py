"""Real owned-process cap controls and portable policy validation."""

import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from complete_lmm_scaling import audit_command  # noqa: E402
from lmm_resource_caps import (  # noqa: E402
    CappedGroup,
    Caps,
    acceptance_failures,  # noqa: E402
    disk_sample,
    monitor,
    run_capped,
    violation,
)


class PolicyTests(unittest.TestCase):
    def test_contradictory_caps_receipts_fail_closed(self):
        caps = Caps().record()
        valid = {
            "caps": caps,
            "final_disk": {"bytes": 100, "free_bytes": 2 * 1024**3},
            "resource_guard": {
                "abort_reason": None,
                "samples": 2,
                "maximum_sampled_job_commit_bytes": 100,
                "os_peak_job_memory_bytes": 100,
                "maximum_sampled_disk_bytes": 100,
                "active_phase_seconds": 0.1,
                "largest_sample_gap_seconds": 0.05,
                "minimum_sampled_free_bytes": 2 * 1024**3,
                "final_active_job_processes": 0,
                "job_events": [],
            },
        }
        self.assertEqual(acceptance_failures(valid, caps), [])
        for field, value in (
            ("active_phase_seconds", 901),
            ("job_events", [{"message": 10}]),
            ("final_active_job_processes", 1),
            ("largest_sample_gap_seconds", None),
        ):
            changed = copy.deepcopy(valid)
            changed["resource_guard"][field] = value
            self.assertTrue(acceptance_failures(changed, caps), field)
        changed = copy.deepcopy(valid)
        changed["final_disk"]["free_bytes"] = 0
        self.assertTrue(acceptance_failures(changed, caps))

    def test_invalid_caps_and_each_guard_are_explicit(self):
        for config in (
            {"job_commit_bytes": True},
            {"seconds": float("nan")},
            {"disk_bytes": 0},
            {"min_free_bytes": -1},
            {"poll_seconds": 2},
        ):
            with self.assertRaises(ValueError):
                Caps(**config)
        caps = Caps(seconds=1, disk_bytes=100, min_free_bytes=10)
        for elapsed, disk, event, expected in (
            (2, {"bytes": 0, "free_bytes": 20}, False, "phase_time_cap"),
            (0, {"bytes": 101, "free_bytes": 20}, False, "sampled_disk_cap"),
            (0, {"bytes": 0, "free_bytes": 9}, False, "free_space_reserve"),
            (0, {"bytes": 0, "free_bytes": 20}, True, "job_memory_limit_event"),
        ):
            self.assertEqual(violation(caps, elapsed, disk, event), expected)


@unittest.skipUnless(os.name == "nt", "reviewed hard-memory job backend requires Windows")
class WindowsCapTests(unittest.TestCase):
    def test_successful_root_with_live_descendant_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            result = run_capped(
                [
                    sys.executable,
                    "-c",
                    "import subprocess,sys;"
                    "subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])",
                ],
                Caps(),
                folder,
                folder / "receipts",
                "orphan",
            )
            self.assertEqual(result["exit_codes"], [0])
            self.assertEqual(
                result["resource_guard"]["abort_reason"], "root_exit_with_live_descendants"
            )
            self.assertGreater(result["resource_guard"]["final_active_job_processes"], 0)

    def test_completed_command_binds_sources_inputs_outputs_and_rejects_substitution(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source, target = folder / "source.py", folder / "output.json"
            source.write_text(
                "from pathlib import Path;import sys,time;time.sleep(.1);"
                "Path(sys.argv[1]).write_text('saved')"
            )
            data = folder / "manifest.json"
            data.write_text("frozen")
            args = [sys.executable, str(source), str(target)]
            inputs, outputs = {"manifest": data}, {"output": target}
            run_capped(args, Caps(), folder, folder / "receipts", "analysis", inputs, outputs)
            audit_command(folder, folder / "receipts", "analysis", args, inputs, outputs, Caps())
            with self.assertRaises(ValueError):
                audit_command(
                    folder,
                    folder / "receipts",
                    "analysis",
                    [sys.executable, "-c", "pass"],
                    inputs,
                    outputs,
                    Caps(),
                )
            target.write_text("changed")
            with self.assertRaises(ValueError):
                audit_command(
                    folder, folder / "receipts", "analysis", args, inputs, outputs, Caps()
                )
            target.write_text("saved")
            data.write_text("changed")
            with self.assertRaises(ValueError):
                audit_command(
                    folder, folder / "receipts", "analysis", args, inputs, outputs, Caps()
                )

    def test_time_cap_terminates_owned_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            result = run_capped(
                [sys.executable, "-c", "import time;time.sleep(30)"],
                Caps(seconds=0.15),
                folder,
                folder / "receipts",
                "time",
            )
            self.assertEqual(result["resource_guard"]["abort_reason"], "phase_time_cap")
            self.assertNotEqual(result["exit_codes"], [0])

    def test_disk_cap_terminates_writer_and_preserves_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            code = (
                "import sys,time;from pathlib import Path;"
                "Path(sys.argv[1]).write_bytes(b'x'*1048576);time.sleep(30)"
            )
            result = run_capped(
                [sys.executable, "-c", code, str(folder / "retained.bin")],
                Caps(disk_bytes=65536),
                folder,
                folder / "receipts",
                "disk",
            )
            self.assertEqual(result["resource_guard"]["abort_reason"], "sampled_disk_cap")
            self.assertEqual((folder / "retained.bin").stat().st_size, 1048576)

    def test_nested_native_child_cannot_allocate_past_hard_job_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            child = (
                "try:\n bytearray(512*1024**2)\nexcept MemoryError:\n "
                "print('allocation_denied',flush=True)\n raise SystemExit(91)\nraise SystemExit(92)"
            )
            root = """import sys
sys.path.insert(0,sys.argv[1])
from lmm_random_slope_stream import ProcessGroup
group=ProcessGroup()
try:
 p=group.start([sys.executable,'-c',sys.argv[2]])
 raise SystemExit(p.wait())
finally: group.close()
"""
            caps = Caps(job_commit_bytes=128 * 1024**2)
            result = run_capped(
                [sys.executable, "-c", root, str(Path(__file__).resolve().parents[1]), child],
                caps,
                folder,
                folder / "receipts",
                "memory",
            )
            self.assertNotEqual(result["exit_codes"], [0])
            logs = "".join(p.read_text() for p in (folder / "receipts").glob("*.log"))
            self.assertTrue(
                "allocation_denied" in logs
                or result["resource_guard"]["abort_reason"] == "job_memory_limit_event"
            )
            self.assertLessEqual(
                result["resource_guard"]["maximum_sampled_job_commit_bytes"], caps.job_commit_bytes
            )

    def test_successful_small_job_records_hard_limit_and_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            caps = Caps()
            group = CappedGroup(caps)
            try:
                group.start([sys.executable, "-c", "import time;time.sleep(.1)"])
                result = monitor(group, caps, folder)
                for p in group.processes:
                    self.assertEqual(p.wait(timeout=10), 0)
                self.assertIsNone(result["abort_reason"])
                self.assertGreater(result["maximum_sampled_job_commit_bytes"], 0)
                self.assertGreater(disk_sample(folder)["free_bytes"], 0)
            finally:
                group.close()


if __name__ == "__main__":
    unittest.main()
