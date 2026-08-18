#!/usr/bin/env python3
"""Replay the released TTT-Discover AHC058 solver on public ALE inputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from evaluate_ahc039_released_public import CPP_FLAGS, case_seed, run_case


REPO_ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_ROOT = Path(os.environ.get("TTT_OFFICIAL_ROOT", REPO_ROOT.parent / "ttt-discover-official"))
STORAGE_ROOT = Path(os.environ.get("TTT_STORAGE_ROOT", REPO_ROOT.parent / "ttt-storage"))
AHC_CACHE = Path(os.environ.get("TTT_AHC_CACHE_DIR", STORAGE_ROOT / "ahc-cache/extracted/cache"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=OFFICIAL_ROOT / "results/algorithm-design/ahc058.cpp",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=AHC_CACHE,
    )
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    parser.add_argument("--max-cases", type=int, default=150)
    parser.add_argument("--candidate-timeout", type=float, default=3.5)
    parser.add_argument("--tester-timeout", type=float, default=30.0)
    parser.add_argument("--compile-timeout", type=float, default=120.0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "docs/ahc058_released_public_150.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "docs/ahc058_released_public_150.csv",
    )
    parser.add_argument(
        "--work-root", type=Path, default=STORAGE_ROOT / "tmp"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tester = args.cache / "tester_binaries/ahc058_tester"
    inputs_dir = args.cache / "public_inputs_150/ahc058_inputs"
    inputs = sorted(inputs_dir.glob("ahc058_*_input.txt"), key=case_seed)
    if args.max_cases > 0:
        inputs = inputs[: args.max_cases]
    for path, label in ((args.source, "released source"), (tester, "official tester")):
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")
    if len(inputs) != args.max_cases:
        raise SystemExit(f"expected {args.max_cases} public inputs, found {len(inputs)}")

    args.work_root.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    wall_started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="ahc058_public_", dir=args.work_root) as tmp_name:
        tmp = Path(tmp_name)
        executable = tmp / "ahc058_released"
        compile_started = time.monotonic()
        compiled = subprocess.run(
            ["g++", *CPP_FLAGS, str(args.source), "-o", str(executable)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.compile_timeout,
            check=False,
        )
        compile_seconds = time.monotonic() - compile_started
        if compiled.returncode != 0:
            raise SystemExit(compiled.stderr.decode("utf-8", "replace")[-4000:])

        output_dir = tmp / "outputs"
        output_dir.mkdir()
        results: list[dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    run_case,
                    executable,
                    tester,
                    input_path,
                    output_dir,
                    args.candidate_timeout,
                    args.tester_timeout,
                )
                for input_path in inputs
            ]
            for completed, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if completed % 25 == 0 or completed == len(futures):
                    accepted = sum(item["status"] == "AC" for item in results)
                    print(f"completed={completed}/{len(futures)} AC={accepted}", flush=True)

    results.sort(key=lambda item: int(item["seed"]))
    scores = [int(item["score"]) for item in results if item["status"] == "AC"]
    status_counts: dict[str, int] = {}
    for item in results:
        status = str(item["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    all_valid = len(scores) == len(results)
    summary = {
        "task": "AHC058 released solver public replay",
        "scope": f"officially generated public seeds 0..{len(results) - 1} with official tester",
        "disclaimer": "Public proxy; not equivalent to the AtCoder hidden-test score in the paper.",
        "source": str(args.source),
        "tester": str(tester),
        "workers": args.workers,
        "host_logical_cpus": os.cpu_count(),
        "num_cases": len(results),
        "accepted": len(scores),
        "status_counts": status_counts,
        "all_cases_valid": all_valid,
        "official_all_or_nothing_score": sum(scores) if all_valid else 0,
        "aggregate_score_ac_only": sum(scores),
        "mean_score": statistics.fmean(scores) if scores else None,
        "median_score": statistics.median(scores) if scores else None,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "compile_seconds": compile_seconds,
        "wall_seconds": time.monotonic() - wall_started,
        "sum_case_seconds": sum(float(item["seconds"]) for item in results),
        "cases": results,
    }
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    with args.output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("seed", "status", "score", "seconds", "returncode", "message"),
            lineterminator="\n",
        )
        writer.writeheader()
        for item in results:
            writer.writerow(item)

    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))
    print(f"json={args.output_json}")
    print(f"csv={args.output_csv}")
    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
