#!/usr/bin/env python3
"""Replay the released TTT-Discover AHC039 solver on 150 public cases.

This uses ALE-Bench's released, officially generated inputs and official tester.
It is a public-training-set replay, not the hidden AtCoder submission score.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SCORE_RE = re.compile(r"Score\s*=\s*(-?\d+)")
REPO_ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_ROOT = Path(os.environ.get("TTT_OFFICIAL_ROOT", REPO_ROOT.parent / "ttt-discover-official"))
STORAGE_ROOT = Path(os.environ.get("TTT_STORAGE_ROOT", REPO_ROOT.parent / "ttt-storage"))
AHC_CACHE = Path(os.environ.get("TTT_AHC_CACHE_DIR", STORAGE_ROOT / "ahc-cache/extracted/cache"))
CPP_FLAGS = (
    "-std=gnu++20",
    "-O2",
    "-DONLINE_JUDGE",
    "-DATCODER",
    "-Wall",
    "-Wextra",
    "-mtune=native",
    "-march=native",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=OFFICIAL_ROOT / "results/algorithm-design/ahc039.cpp",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=AHC_CACHE,
    )
    # The released solver has a randomized empty-rectPool SIGFPE path. Higher
    # concurrency makes it much more frequent, so exact-artifact replay defaults
    # to one case at a time even on the 128-logical-CPU host.
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=150)
    parser.add_argument(
        "--seeds",
        help="Optional comma-separated seed subset, for diagnosing or retrying cases.",
    )
    parser.add_argument("--candidate-timeout", type=float, default=3.5)
    parser.add_argument("--tester-timeout", type=float, default=30.0)
    parser.add_argument("--compile-timeout", type=float, default=120.0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "docs/ahc039_released_public_150.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "docs/ahc039_released_public_150.csv",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=STORAGE_ROOT / "tmp",
    )
    return parser.parse_args()


def case_seed(path: Path) -> int:
    return int(path.stem.split("_")[-2])


def run_case(
    executable: Path,
    tester: Path,
    input_path: Path,
    output_dir: Path,
    candidate_timeout: float,
    tester_timeout: float,
) -> dict[str, object]:
    started = time.monotonic()
    output_path = output_dir / f"{input_path.stem}.out"
    seed = case_seed(input_path)
    try:
        with input_path.open("rb") as stdin:
            candidate = subprocess.run(
                [str(executable)],
                stdin=stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=candidate_timeout,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return {"seed": seed, "status": "TLE", "score": None, "seconds": time.monotonic() - started}

    if candidate.returncode != 0:
        return {
            "seed": seed,
            "status": "RE",
            "score": None,
            "returncode": candidate.returncode,
            "seconds": time.monotonic() - started,
            "message": (
                candidate.stderr.decode("utf-8", "replace")[-500:]
                or f"candidate exited with return code {candidate.returncode}"
            ),
        }

    output_path.write_bytes(candidate.stdout)
    try:
        checked = subprocess.run(
            [str(tester), str(input_path), str(output_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=tester_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"seed": seed, "status": "TESTER_TLE", "score": None, "seconds": time.monotonic() - started}

    judge_text = (checked.stdout + checked.stderr).decode("utf-8", "replace")
    match = SCORE_RE.search(judge_text)
    if checked.returncode != 0 or match is None:
        return {
            "seed": seed,
            "status": "WA",
            "score": None,
            "seconds": time.monotonic() - started,
            "message": judge_text[-500:],
        }
    return {
        "seed": seed,
        "status": "AC",
        "score": int(match.group(1)),
        "seconds": time.monotonic() - started,
    }


def main() -> int:
    args = parse_args()
    tester = args.cache / "tester_binaries/ahc039_tester"
    inputs_dir = args.cache / "public_inputs_150/ahc039_inputs"
    inputs = sorted(inputs_dir.glob("ahc039_*_input.txt"), key=case_seed)
    if args.seeds:
        selected = {int(value) for value in args.seeds.split(",")}
        inputs = [path for path in inputs if case_seed(path) in selected]
        if {case_seed(path) for path in inputs} != selected:
            raise SystemExit(f"not all requested seeds exist: {sorted(selected)}")
    elif args.max_cases > 0:
        inputs = inputs[: args.max_cases]
    if not args.source.is_file():
        raise SystemExit(f"released source not found: {args.source}")
    if not tester.is_file():
        raise SystemExit(f"official tester not found: {tester}")
    if not args.seeds and len(inputs) != args.max_cases:
        raise SystemExit(f"expected {args.max_cases} public inputs, found {len(inputs)} in {inputs_dir}")

    args.work_root.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    wall_started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="ahc039_public_", dir=args.work_root) as tmp_name:
        tmp = Path(tmp_name)
        executable = tmp / "ahc039_released"
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
            futures = {
                pool.submit(
                    run_case,
                    executable,
                    tester,
                    input_path,
                    output_dir,
                    args.candidate_timeout,
                    args.tester_timeout,
                ): input_path
                for input_path in inputs
            }
            for completed, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                if completed % 25 == 0 or completed == len(futures):
                    accepted = sum(item["status"] == "AC" for item in results)
                    print(f"completed={completed}/{len(futures)} AC={accepted}", flush=True)

    results.sort(key=lambda item: int(item["seed"]))
    scores = [int(item["score"]) for item in results if item["status"] == "AC"]
    status_counts: dict[str, int] = {}
    for item in results:
        status = str(item["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    wall_seconds = time.monotonic() - wall_started
    summary = {
        "task": "AHC039 released solver public replay",
        "scope": (
            f"officially generated public seeds {results[0]['seed']}..{results[-1]['seed']} "
            "with official tester"
        ),
        "disclaimer": "Not equivalent to the AtCoder hidden-test score reported in the paper.",
        "source": str(args.source),
        "tester": str(tester),
        "workers": args.workers,
        "host_logical_cpus": os.cpu_count(),
        "num_cases": len(results),
        "accepted": len(scores),
        "status_counts": status_counts,
        "all_cases_valid": len(scores) == len(results),
        "official_all_or_nothing_score": sum(scores) if len(scores) == len(results) else 0,
        "aggregate_score_ac_only": sum(scores),
        "mean_score": statistics.fmean(scores) if scores else None,
        "median_score": statistics.median(scores) if scores else None,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "compile_seconds": compile_seconds,
        "wall_seconds": wall_seconds,
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
    return 0 if len(scores) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
