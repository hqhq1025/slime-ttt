"""Aggregate per-artifact MLA Decode A100 proxy results into JSON and CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()

    artifacts = []
    rows = []
    for path in args.inputs:
        result = json.loads(path.read_text())
        artifact = path.stem.removeprefix("result-full-")
        artifacts.append({"artifact": artifact, "result_file": str(path), **result})
        for case in result.get("case_results") or []:
            rows.append(
                {
                    "artifact": artifact,
                    "kind": case["kind"],
                    "prefill": case["prefill"],
                    "seed": case["seed"],
                    "ok": case["ok"],
                    "runs": case["runs"],
                    "mean_us": case["mean_us"],
                    "max_abs_error": case["max_abs_error"],
                    "peak_memory_gib": case["peak_memory_gib"],
                    "elapsed_s": case["elapsed_s"],
                    "error": case["error"],
                }
            )

    report = {
        "hardware_adaptation": "NVIDIA A100 proxy",
        "comparability": "not comparable to official H200/MI300X results",
        "artifacts": artifacts,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n")

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "artifact",
        "kind",
        "prefill",
        "seed",
        "ok",
        "runs",
        "mean_us",
        "max_abs_error",
        "peak_memory_gib",
        "elapsed_s",
        "error",
    ]
    with args.csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
