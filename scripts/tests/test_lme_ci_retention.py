"""The standalone CI runner must use the same retention as Task."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "ci" / "lme_ci.py"
SPEC = importlib.util.spec_from_file_location("lme_ci_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


class RunnerRetentionTests(unittest.TestCase):
    def test_direct_ci_runner_enters_retention_once(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(runner.subprocess, "call", return_value=17) as call:
                for command in (["ci", "--skip-python"], ["preflight"], ["windows-lld-ci"]):
                    with self.subTest(command=command):
                        self.assertEqual(runner.main(command), 17)
                        self.assertEqual(
                            call.call_args.kwargs["env"]["LME_CI_RETENTION_CHILD"], "1"
                        )
                        self.assertEqual(call.call_args.args[0][-len(command) :], command)
            with mock.patch.dict(os.environ, {"LME_CI_RETENTION_CHILD": "1"}):
                with mock.patch.object(runner, "ci") as ci:
                    self.assertEqual(runner.main(["ci", "--skip-python"]), 0)
                    ci.assert_called_once_with(reuse_venv=False, skip_wheel=False, skip_python=True)
