"""Evaluate a denoising candidate with the released TTT-Discover contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path


BASELINES = {
    "mse": {"baseline": 0.304721, "perfect": 0.0},
    "poisson": {"baseline": 0.257575, "perfect": 0.031739},
}


def normalized(value: float, metric: str) -> float:
    bounds = BASELINES[metric]
    return (bounds["baseline"] - value) / (bounds["baseline"] - bounds["perfect"])


def load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location("ttt_denoising_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "magic_denoise", None)
    if not callable(function):
        raise TypeError(f"{path} does not define callable magic_denoise")
    return function


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from examples.denoising.utils import run_denoising_eval

    start = time.monotonic()
    mse, poisson = run_denoising_eval(load_candidate(args.candidate.resolve()), seed=args.seed)
    mse_norm = normalized(float(mse), "mse")
    poisson_norm = normalized(float(poisson), "poisson")
    result = {
        "candidate": str(args.candidate.resolve()),
        "seed": args.seed,
        "mse": float(mse),
        "poisson": float(poisson),
        "mse_normalized": mse_norm,
        "poisson_normalized": poisson_norm,
        "mean_normalized": (mse_norm + poisson_norm) / 2,
        "valid": poisson_norm >= 0.97,
        "elapsed_seconds": time.monotonic() - start,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
