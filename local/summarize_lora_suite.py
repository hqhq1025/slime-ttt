#!/usr/bin/env python3
"""Summarize full-parameter vs LoRA TTT runs from checkpoint artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RUNS = {
    "erdos": ("qwen3-4b-ttt-erdos-8x16x10", "qwen3-4b-ttt-erdos-lora-r32-8x16x10", -1),
    "ac1": ("qwen3-4b-ttt-ac1-8x16x5", "qwen3-4b-ttt-ac1-lora-r32-8x16x5", -1),
    "ac2": ("qwen3-4b-ttt-ac2-8x16x5", "qwen3-4b-ttt-ac2-lora-r32-8x16x5", 1),
    "circle26": ("qwen3-4b-ttt-circle26-8x16x5", "qwen3-4b-ttt-circle26-lora-r32-8x16x5", 1),
    "circle32": ("qwen3-4b-ttt-circle32-8x16x5", "qwen3-4b-ttt-circle32-lora-r32-8x16x5", 1),
    "trimul": ("qwen3-4b-ttt-trimul-a100-8x8x3", "qwen3-4b-ttt-trimul-a100-lora-r32-8x8x3", -1),
    "mla": (
        "qwen3-4b-ttt-mla-a100-full-pair-2step",
        "qwen3-4b-ttt-mla-a100-lora-r32-pair-2step-graphfix",
        -1,
    ),
    "ahc039": ("qwen3-4b-ttt-ahc039-full-pair-2step", "qwen3-4b-ttt-ahc039-lora-r32-pair-2step", 1),
    "ahc058": ("qwen3-4b-ttt-ahc058-full-pair-2step", "qwen3-4b-ttt-ahc058-lora-r32-pair-2step", 1),
    "denoising": ("qwen3-4b-ttt-denoising-full-pair-2step", "qwen3-4b-ttt-denoising-lora-r32-pair-2step", -1),
}


def summarize(path: Path, raw_sign: int) -> dict:
    valid_by_step = []
    rollout_steps = []
    total = 0
    truncated = 0
    for file in sorted((path / "ttt_rollouts").glob("rollout_*.json")):
        # Shared filesystems can briefly retain a directory entry whose inode
        # is already gone. Treat it as a missing artifact instead of failing
        # the complete cross-domain report.
        try:
            samples = json.loads(file.read_text())
        except FileNotFoundError:
            continue
        rollout_steps.append(int(file.stem.rsplit("_", 1)[-1]))
        total += len(samples)
        valid_by_step.append(
            sum(float(sample.get("metadata", {}).get("correctness", 0)) == 1 for sample in samples)
        )
        truncated += sum(sample.get("status") == "truncated" for sample in samples)

    best_curve = []
    archive_steps = []
    archive_sizes = []
    for file in sorted((path / "ttt_archive").glob("archive_step_*.json")):
        archive = json.loads(file.read_text())
        archive_steps.append(int(file.stem.rsplit("_", 1)[-1]))
        values = [
            state["value"]
            for state in archive.get("states", [])
            if isinstance(state.get("value"), (int, float))
        ]
        best_curve.append(raw_sign * max(values))
        archive_sizes.append(len(archive.get("states", [])))

    return {
        "exists": path.is_dir(),
        "steps": len(best_curve),
        "samples": total,
        "valid": sum(valid_by_step),
        "valid_rate": sum(valid_by_step) / total if total else None,
        "valid_by_step": valid_by_step,
        "rollout_steps": rollout_steps,
        "missing_rollout_steps": sorted(set(archive_steps) - set(rollout_steps)),
        "truncated": truncated,
        "best_curve": best_curve,
        "archive_steps": archive_steps,
        "final_best": best_curve[-1] if best_curve else None,
        "archive_sizes": archive_sizes,
        "final_archive_size": archive_sizes[-1] if archive_sizes else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {}
    for task, (full_name, lora_name, raw_sign) in RUNS.items():
        result[task] = {
            "full": summarize(args.checkpoints / full_name, raw_sign),
            "lora": summarize(args.checkpoints / lora_name, raw_sign),
        }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
