"""Independent serial correctness, complete-operation timing, and memory checks."""

import argparse
import hashlib
import json
import platform
import random
import statistics
import sys
import time
import tracemalloc
import types
from pathlib import Path

sys.dont_write_bytecode = True
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True, help="New JSON receipt path")
args = parser.parse_args()
destination = args.output

root = Path(__file__).resolve().parent
sources = {
    name: (root / "pipeline" / name / "pipeline.py").read_bytes()
    for name in ("original", "baseline", "current", "compact", "compact-v2", "final")
}
modules = {}
for name, source in sources.items():
    module = types.ModuleType("trial_" + name.replace("-", "_"))
    exec(compile(source, f"<{name}-pipeline>", "exec"), module.__dict__)
    modules[name] = module


def outcome(function, payload, keys):
    try:
        return ("ok", function(payload, keys))
    except Exception as error:
        return ("error", type(error).__name__)


rng = random.Random(2106)
cases = [
    (
        '[{"category":"a","value":1e16},{"category":"a","value":1},{"category":"a","value":-1e16}]',
        ["a"],
    ),
    ('[{"category":"unrequested"}]', ["a"]),
    ("{", []),
    ("{", ["a"]),
    ("[]", []),
    ("[]", ["missing"]),
]
for _ in range(500):
    rows = []
    for _ in range(rng.randrange(61)):
        record = {"category": rng.choice(["a", "b", "c", "unused"])}
        if record["category"] != "unused" or rng.randrange(3):
            record["value"] = rng.choice([-1e16, 1e16, -3, 0, 1, 0.125, 7.5])
        rows.append(record)
    cases.append((json.dumps(rows), rng.sample(["a", "b", "c", "missing"], rng.randrange(5))))

correctness = {}
for name, module in modules.items():
    failures = []
    for number, (payload, keys) in enumerate(cases):
        expected = outcome(modules["original"].totals, payload, keys)
        observed = outcome(module.totals, payload, keys)
        if observed != expected:
            failures.append(
                {"case": number, "expected": repr(expected), "observed": repr(observed)}
            )
    correctness[name] = {"cases": len(cases), "passed": not failures, "failures": failures}

workloads = [
    ("complete_6000x45", 6000, 45),
    ("adjacent_2000x1", 2000, 1),
    ("adjacent_2000x0", 2000, 0),
]
measurements = {}
for label, rows, key_count in workloads:
    payload = json.dumps([{"category": str(i % 45), "value": i % 11} for i in range(rows)])
    keys = [str(i) for i in range(key_count)]
    expected = modules["original"].totals(payload, keys)
    samples = {name: [] for name in modules}
    for block in range(5):
        order = list(modules)
        if block % 2:
            order.reverse()
        for name in order:
            function = modules[name].totals
            for _ in range(2):
                assert function(payload, keys) == expected
            for _ in range(11):
                start = time.perf_counter()
                observed = function(payload, keys)
                elapsed = time.perf_counter() - start
                assert observed == expected
                samples[name].append(elapsed)
    results = {}
    for name, function_module in modules.items():
        peaks = []
        for _ in range(3):
            tracemalloc.start()
            observed = function_module.totals(payload, keys)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            assert observed == expected
            peaks.append(peak)
        results[name] = {
            "samples_seconds": samples[name],
            "count": len(samples[name]),
            "median_seconds": statistics.median(samples[name]),
            "range_seconds": [min(samples[name]), max(samples[name])],
            "tracemalloc_peak_bytes": peaks,
        }
    original_median = results["original"]["median_seconds"]
    for result in results.values():
        result["original_to_candidate_median_ratio"] = original_median / result["median_seconds"]
    measurements[label] = {
        "records": rows,
        "requested_categories": key_count,
        "input_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "results": results,
    }

document = {
    "python": sys.version,
    "platform": platform.platform(),
    "source_sha256": {name: hashlib.sha256(source).hexdigest() for name, source in sources.items()},
    "correctness": correctness,
    "measurements": measurements,
    "protocol": {
        "blocks": 5,
        "warmups_per_block": 2,
        "measured_samples_per_block": 11,
        "order": "forward/reverse alternating blocks",
        "timing": "serialized complete operation; correctness outside elapsed measurement",
        "memory": "separate tracemalloc runs; Python allocation peak, not process RSS",
    },
    "limits": (
        "One agent-produced implementation per arm; observations are not a causal estimate "
        "of model/skill effectiveness or total agent latency. The 506-case differential suite "
        "is a held-out stress check added after the first discovery runs, "
        "not a preregistered primary endpoint."
    ),
}
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(document, indent=2) + "\n")
print(
    json.dumps(
        {
            "correctness": correctness,
            "timings": {
                label: {
                    name: {k: v for k, v in result.items() if k not in ("samples_seconds",)}
                    for name, result in group["results"].items()
                }
                for label, group in measurements.items()
            },
            "output": str(destination.resolve()),
        },
        indent=2,
    )
)
