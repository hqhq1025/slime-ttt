"""Evaluate a denoising artifact on the official PBMC/Tabula held-out protocol."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
STORAGE_ROOT = Path(os.environ.get("TTT_STORAGE_ROOT", REPO_ROOT.parent / "ttt-storage"))
OFFICIAL_ROOT = Path(os.environ.get("TTT_OFFICIAL_ROOT", REPO_ROOT.parent / "ttt-discover-official"))
OPENPROBLEMS_ROOT = Path(os.environ.get("OPENPROBLEMS_ROOT", STORAGE_ROOT / "openproblems-v1.0.0"))


def load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location("ttt_denoising_heldout_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "magic_denoise", None)
    if not callable(function):
        raise TypeError(f"{path} does not define callable magic_denoise")
    return function


def normalized(score: float, no_denoise_score: float, perfect_score: float) -> float:
    return (no_denoise_score - score) / (no_denoise_score - perfect_score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("pbmc", "tabula"))
    parser.add_argument(
        "--candidate",
        type=Path,
        default=OFFICIAL_ROOT / "results/denoising/denoise_ttt.py",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Keep this evaluator isolated from slime's own ``examples`` package.
    sys.path.insert(0, str(OPENPROBLEMS_ROOT))
    os.environ.setdefault("OPENPROBLEMS_CACHE_DIR", str(STORAGE_ROOT / "openproblems_cache"))
    os.environ.setdefault(
        "OPENPROBLEMS_PBMC_PATH",
        str(STORAGE_ROOT / "datasets/openproblems/heldout/pbmc.h5ad"),
    )
    os.environ.setdefault(
        "OPENPROBLEMS_TABULA_LUNG_PATH",
        str(STORAGE_ROOT / "datasets/openproblems/heldout/tabula-muris-senis-lung.h5ad"),
    )

    import openproblems.data
    import scprep
    from openproblems.tasks.denoising.datasets.pbmc import pbmc
    from openproblems.tasks.denoising.datasets.tabula_muris_senis import (
        tabula_muris_senis_lung_random,
    )
    from openproblems.tasks.denoising.methods.baseline import no_denoising, perfect_denoising
    from openproblems.tasks.denoising.metrics.mse import mse
    from openproblems.tasks.denoising.metrics.poisson import poisson

    # OPENPROBLEMS_CACHE_DIR is persistent project storage, so do not remove it at exit.
    openproblems.data.no_cleanup()
    started = time.monotonic()
    loader = pbmc if args.dataset == "pbmc" else tabula_muris_senis_lung_random
    adata = loader(test=False)

    none = no_denoising(adata.copy())
    perfect = perfect_denoising(adata.copy())
    mse_none, mse_perfect = float(mse(none)), float(mse(perfect))
    poisson_none, poisson_perfect = float(poisson(none)), float(poisson(perfect))

    denoised = load_candidate(args.candidate.resolve())(
        scprep.utils.toarray(adata.obsm["train"]), verbose=args.verbose
    )
    evaluated = adata.copy()
    evaluated.obsm["denoised"] = denoised
    mse_value, poisson_value = float(mse(evaluated)), float(poisson(evaluated))
    mse_norm = normalized(mse_value, mse_none, mse_perfect)
    poisson_norm = normalized(poisson_value, poisson_none, poisson_perfect)

    result = {
        "dataset": args.dataset,
        "shape": list(adata.shape),
        "candidate": str(args.candidate.resolve()),
        "mse": mse_value,
        "poisson": poisson_value,
        "mse_no_denoising": mse_none,
        "mse_perfect": mse_perfect,
        "poisson_no_denoising": poisson_none,
        "poisson_perfect": poisson_perfect,
        "mse_normalized": mse_norm,
        "poisson_normalized": poisson_norm,
        "mean_normalized": (mse_norm + poisson_norm) / 2,
        "valid": bool(poisson_norm >= 0.97),
        "elapsed_seconds": time.monotonic() - started,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
