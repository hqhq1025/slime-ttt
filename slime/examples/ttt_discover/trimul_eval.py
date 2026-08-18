"""Local A100 evaluator for the official GPU Mode TriMul task.

The cases, input distribution, reference operator, tolerances, CUDA-event
timing, and geometric-mean ranking match ``examples/gpu_mode/lib/bioml/trimul``
from the TTT-Discover repository.  The evaluator deliberately runs candidate
code in a fresh process: Triton modules and CUDA contexts left by one candidate
must not affect the next candidate.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


OFFICIAL_TEST_CASES = [
    dict(seqlen=32, bs=1, dim=128, hiddendim=128, seed=9371, nomask=True, distribution="normal"),
    dict(seqlen=32, bs=1, dim=128, hiddendim=128, seed=1092, nomask=False, distribution="normal"),
    dict(seqlen=64, bs=2, dim=256, hiddendim=128, seed=2291, nomask=True, distribution="normal"),
    dict(seqlen=64, bs=2, dim=256, hiddendim=128, seed=210284, nomask=False, distribution="normal"),
    dict(seqlen=128, bs=1, dim=768, hiddendim=128, seed=81934, nomask=True, distribution="normal"),
    dict(seqlen=256, bs=1, dim=128, hiddendim=128, seed=1932, nomask=True, distribution="normal"),
    dict(seqlen=256, bs=1, dim=128, hiddendim=128, seed=10432, nomask=False, distribution="normal"),
    dict(seqlen=768, bs=2, dim=128, hiddendim=128, seed=731, nomask=True, distribution="normal"),
    dict(seqlen=1024, bs=1, dim=384, hiddendim=128, seed=53121, nomask=False, distribution="normal"),
    dict(seqlen=1024, bs=1, dim=768, hiddendim=128, seed=31, nomask=True, distribution="normal"),
    dict(seqlen=1024, bs=1, dim=768, hiddendim=128, seed=4921, nomask=False, distribution="normal"),
    dict(seqlen=32, bs=1, dim=128, hiddendim=128, seed=937321, nomask=True, distribution="cauchy"),
    dict(seqlen=64, bs=2, dim=256, hiddendim=128, seed=2291, nomask=True, distribution="cauchy"),
    dict(seqlen=128, bs=1, dim=768, hiddendim=128, seed=8134, nomask=True, distribution="cauchy"),
    dict(seqlen=256, bs=1, dim=128, hiddendim=128, seed=932, nomask=True, distribution="cauchy"),
    dict(seqlen=768, bs=2, dim=128, hiddendim=128, seed=31, nomask=True, distribution="cauchy"),
    dict(seqlen=1024, bs=1, dim=384, hiddendim=128, seed=5321, nomask=False, distribution="cauchy"),
    dict(seqlen=1024, bs=1, dim=768, hiddendim=128, seed=491, nomask=False, distribution="cauchy"),
]

OFFICIAL_BENCHMARK_CASES = [
    dict(seqlen=256, bs=2, dim=128, hiddendim=128, seed=9371, nomask=True, distribution="normal"),
    dict(seqlen=768, bs=1, dim=128, hiddendim=128, seed=381, nomask=True, distribution="cauchy"),
    dict(seqlen=256, bs=2, dim=384, hiddendim=128, seed=2301, nomask=False, distribution="normal"),
    dict(seqlen=512, bs=1, dim=128, hiddendim=128, seed=12819, nomask=True, distribution="normal"),
    dict(seqlen=1024, bs=1, dim=128, hiddendim=128, seed=381, nomask=True, distribution="cauchy"),
    dict(seqlen=768, bs=1, dim=384, hiddendim=128, seed=481, nomask=False, distribution="normal"),
    dict(seqlen=1024, bs=1, dim=384, hiddendim=128, seed=23291, nomask=True, distribution="normal"),
]

# Both mask branches, both distributions, and two channel sizes.  The timing
# cases are official leaderboard cases rather than toy-only shapes.
SMOKE_TEST_INDICES = (0, 1, 4, 11)
SMOKE_BENCHMARK_INDICES = (0, 2)


@dataclass
class EvalSummary:
    ok: bool
    score_us: float = 0.0
    passed: int = 0
    total: int = 0
    benchmark_means_us: list[float] | None = None
    max_abs_error: float = 0.0
    error: str = ""
    elapsed_s: float = 0.0


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def cases_for_profile(profile: str):
    if profile == "full":
        return list(OFFICIAL_TEST_CASES), list(OFFICIAL_BENCHMARK_CASES)
    if profile == "smoke":
        return (
            [OFFICIAL_TEST_CASES[i] for i in SMOKE_TEST_INDICES],
            [OFFICIAL_BENCHMARK_CASES[i] for i in SMOKE_BENCHMARK_INDICES],
        )
    raise ValueError(f"unknown TriMul profile: {profile}")


def _load_candidate(path: str):
    spec = importlib.util.spec_from_file_location("ttt_trimul_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import candidate {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "custom_kernel", None)
    if not callable(fn):
        raise AttributeError("candidate must define callable custom_kernel(data)")
    return fn


def _generate_input(case: dict, torch):
    bs, n, dim, hidden = case["bs"], case["seqlen"], case["dim"], case["hiddendim"]
    generator = torch.Generator(device="cuda").manual_seed(case["seed"])
    if case["distribution"] == "cauchy":
        u = torch.empty((bs, n, n, dim), device="cuda", dtype=torch.float32)
        u.uniform_(0.0, 1.0, generator=generator)
        inp = 2.0 * torch.tan(math.pi * (u - 0.5))
    else:
        inp = torch.randn(
            (bs, n, n, dim), device="cuda", dtype=torch.float32, generator=generator
        ).contiguous()
    if case["nomask"]:
        mask = torch.ones(bs, n, n, device="cuda")
    else:
        mask = torch.randint(0, 2, (bs, n, n), device="cuda", generator=generator)
    weights = {
        "norm.weight": torch.randn(dim, device="cuda", dtype=torch.float32),
        "norm.bias": torch.randn(dim, device="cuda", dtype=torch.float32),
        "left_proj.weight": torch.randn(hidden, dim, device="cuda") / math.sqrt(hidden),
        "right_proj.weight": torch.randn(hidden, dim, device="cuda") / math.sqrt(hidden),
        "left_gate.weight": torch.randn(hidden, dim, device="cuda") / math.sqrt(hidden),
        "right_gate.weight": torch.randn(hidden, dim, device="cuda") / math.sqrt(hidden),
        "out_gate.weight": torch.randn(hidden, dim, device="cuda") / math.sqrt(hidden),
        "to_out_norm.weight": torch.randn(hidden, device="cuda", dtype=torch.float32),
        "to_out.weight": torch.randn(dim, hidden, device="cuda") / math.sqrt(dim),
        "to_out_norm.bias": torch.randn(hidden, device="cuda", dtype=torch.float32),
    }
    # Intentionally do not add ``nomask`` here: this matches the official
    # generate_input exactly.  Kernels must honor the actual mask tensor.
    return inp, mask, weights, {"hidden_dim": hidden, "dim": dim}


def _reference(data, torch):
    import torch.nn.functional as functional

    inp, mask, weights, _ = data
    x = functional.layer_norm(
        inp, (inp.shape[-1],), weights["norm.weight"], weights["norm.bias"]
    )
    left = functional.linear(x, weights["left_proj.weight"])
    right = functional.linear(x, weights["right_proj.weight"])
    left = left * mask.unsqueeze(-1) * torch.sigmoid(
        functional.linear(x, weights["left_gate.weight"])
    )
    right = right * mask.unsqueeze(-1) * torch.sigmoid(
        functional.linear(x, weights["right_gate.weight"])
    )
    out_gate = torch.sigmoid(functional.linear(x, weights["out_gate.weight"]))
    out = torch.einsum("...ikd,...jkd->...ijd", left, right)
    out = functional.layer_norm(
        out, (out.shape[-1],), weights["to_out_norm.weight"], weights["to_out_norm.bias"]
    )
    return functional.linear(out * out_gate, weights["to_out.weight"])


def _clone_data(data, torch):
    inp, mask, weights, config = data
    return inp.clone(), mask.clone(), {key: value.clone() for key, value in weights.items()}, dict(config)


def evaluate_candidate(path: str, profile: str, repeats: int) -> EvalSummary:
    import torch

    started = time.monotonic()
    if not torch.cuda.is_available():
        return EvalSummary(ok=False, error="CUDA is unavailable")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    candidate = _load_candidate(path)
    tests, benchmarks = cases_for_profile(profile)
    max_error = 0.0
    passed = 0
    try:
        for index, case in enumerate(tests):
            data = _generate_input(case, torch)
            expected = _reference(data, torch)
            actual = candidate(_clone_data(data, torch))
            torch.cuda.synchronize()
            if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape:
                raise ValueError(f"test {index}: output shape/type mismatch")
            if not torch.allclose(actual, expected, rtol=2e-2, atol=2e-2, equal_nan=True):
                diff = torch.abs(actual.float() - expected.float())
                mismatch_count = int((diff > (2e-2 + 2e-2 * torch.abs(expected))).sum())
                raise ValueError(f"test {index}: {mismatch_count} values outside tolerance")
            max_error = max(max_error, float(torch.max(torch.abs(actual.float() - expected.float()))))
            passed += 1
            del data, expected, actual
            gc.collect()
            torch.cuda.empty_cache()

        means_us = []
        for index, case in enumerate(benchmarks):
            data = _generate_input(case, torch)
            # Obligatory correctness check before timing, as in official eval.py.
            expected = _reference(data, torch)
            actual = candidate(_clone_data(data, torch))
            torch.cuda.synchronize()
            if not torch.allclose(actual, expected, rtol=2e-2, atol=2e-2, equal_nan=True):
                raise ValueError(f"benchmark {index}: correctness check failed")
            del expected, actual
            for _ in range(2):
                candidate(data)
            torch.cuda.synchronize()
            durations_us = []
            for repeat_index in range(max(3, repeats)):
                # Official leaderboard mode re-generates inputs with seed += 13
                # and rechecks correctness on every timed repetition.
                timed_case = dict(case)
                timed_case["seed"] += 13 * (repeat_index + 1)
                timed_data = _generate_input(timed_case, torch)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = candidate(timed_data)
                end.record()
                torch.cuda.synchronize()
                durations_us.append(float(start.elapsed_time(end) * 1000.0))
                timed_expected = _reference(timed_data, torch)
                if not torch.allclose(
                    output, timed_expected, rtol=2e-2, atol=2e-2, equal_nan=True
                ):
                    raise ValueError(
                        f"benchmark {index}, repeat {repeat_index}: correctness recheck failed"
                    )
                del output, timed_expected, timed_data
            means_us.append(statistics.fmean(durations_us))
            del data
            gc.collect()
            torch.cuda.empty_cache()
        return EvalSummary(
            ok=True,
            score_us=geometric_mean(means_us),
            passed=passed,
            total=len(tests),
            benchmark_means_us=means_us,
            max_abs_error=max_error,
            elapsed_s=time.monotonic() - started,
        )
    except Exception as exc:  # candidate errors are evaluator outcomes
        return EvalSummary(
            ok=False,
            passed=passed,
            total=len(tests),
            max_abs_error=max_error,
            error=f"{type(exc).__name__}: {exc}",
            elapsed_s=time.monotonic() - started,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = evaluate_candidate(args.candidate, args.profile, args.repeats)
    Path(args.output).write_text(json.dumps(asdict(summary), indent=2))
    print(json.dumps(asdict(summary)), flush=True)
    return 0 if summary.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
