"""Single-A100 proxy evaluator for TTT-Discover's official MLA Decode task.

This reuses the official input generator, reference implementation, tolerances,
and entrypoint contract.  The smoke score is the geometric mean over two small
official *correctness* shapes timed locally on an NVIDIA A100.  The ``full``
suite additionally covers all four official correctness shapes and both
official benchmark shapes.  It is not the official MI300X/H200 leaderboard.
"""

from __future__ import annotations

import argparse
import dataclasses
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


DEFAULT_OFFICIAL_ROOT = Path(
    os.environ.get(
        "TTT_OFFICIAL_ROOT",
        Path(__file__).resolve().parents[3].parent / "ttt-discover-official",
    )
)


CORRECTNESS_CASES = [
    dict(batchsize=128, dim=7168, dq=1536, prefill=128, seed=9247),
    dict(batchsize=128, dim=7168, dq=1536, prefill=512, seed=2197),
    dict(batchsize=128, dim=7168, dq=1536, prefill=1024, seed=9107),
    dict(batchsize=128, dim=7168, dq=1536, prefill=2048, seed=5291),
]
BENCHMARK_CASES = [
    dict(batchsize=128, dim=7168, dq=1536, prefill=4096, seed=9817),
    dict(batchsize=128, dim=7168, dq=1536, prefill=6144, seed=5291),
]
SMOKE_CASES = CORRECTNESS_CASES[:2]


@dataclass
class EvalSummary:
    ok: bool
    score_us: float = 0.0
    passed: int = 0
    total: int = 0
    benchmark_means_us: list[float] | None = None
    case_results: list[dict] | None = None
    max_abs_error: float = 0.0
    peak_memory_gib: float = 0.0
    elapsed_s: float = 0.0
    hardware: str = ""
    suite: str = "smoke"
    comparability: str = "A100 proxy; not comparable to official MI300X/H200 results"
    error: str = ""


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _load_official_modules(official_root: Path):
    task_dir = official_root / "examples/gpu_mode/lib/mla-decode"
    if not (task_dir / "reference.py").is_file():
        raise FileNotFoundError(f"official MLA Decode task not found under {task_dir}")
    sys.path.insert(0, str(task_dir))
    import reference  # type: ignore[import-not-found]

    return reference


def _load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location("ttt_mla_decode_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import candidate {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "custom_kernel", None)
    if not callable(fn):
        raise AttributeError("candidate must define callable custom_kernel(data)")
    return fn


def _clone_config(config, dataclasses_module):
    return dataclasses_module.replace(
        config,
        Q_proj_down_weight=config.Q_proj_down_weight.clone(),
        Q_proj_up_weight=config.Q_proj_up_weight.clone(),
        KV_proj_down_weight=config.KV_proj_down_weight.clone(),
        KV_proj_up_weight=config.KV_proj_up_weight.clone(),
        wo_weight=config.wo_weight.clone(),
    )


def _clone_cache(cache, reference):
    copied = reference.KVCache(tuple(cache.data.shape)).cuda()
    copied.load_state_dict(cache.state_dict())
    copied.seq_len = cache.seq_len
    return copied


def _check_pair(actual, expected, torch) -> float:
    actual_out, actual_cache = actual
    expected_out, expected_cache = expected
    if actual_out.shape != expected_out.shape or actual_cache.shape != expected_cache.shape:
        raise ValueError(
            f"shape mismatch: got {(actual_out.shape, actual_cache.shape)}, "
            f"expected {(expected_out.shape, expected_cache.shape)}"
        )
    if not torch.allclose(actual_out, expected_out, rtol=2e-2, atol=8e-3, equal_nan=True):
        raise ValueError("MLA output does not match the official reference")
    max_error = float((actual_out.float() - expected_out.float()).abs().max())

    # Chunk the 1.1-GiB cache comparison so the checker itself does not create
    # a second cache-sized fp32 temporary on an 80-GiB A100.
    for start in range(0, actual_cache.shape[1], 256):
        got = actual_cache[:, start : start + 256]
        want = expected_cache[:, start : start + 256]
        if not torch.allclose(got, want, rtol=2e-2, atol=8e-3, equal_nan=True):
            raise ValueError(f"KV cache mismatch in sequence slice starting at {start}")
        max_error = max(max_error, float((got.float() - want.float()).abs().max()))
    return max_error


def _run_reference_chunked(case: dict, reference, torch, chunk_size: int = 16):
    """Evaluate the unmodified official reference a few batch rows at a time.

    The operation is batch-separable.  Chunking only the batch dimension keeps
    the 6144-token reference below A100 memory while preserving the official
    math, weights, inputs, tolerances, and output contract.
    """
    config, x, cache = reference.generate_input(**case)
    write_index = cache.seq_len
    output_chunks = []
    token_chunks = []
    with torch.no_grad():
        for start in range(0, config.batch_size, chunk_size):
            stop = min(start + chunk_size, config.batch_size)
            chunk_cache = reference.KVCache(
                (stop - start, *cache.data.shape[1:])
            ).cuda()
            chunk_cache.data.copy_(cache.data[start:stop])
            chunk_cache.seq_len = cache.seq_len
            chunk_config = dataclasses.replace(
                config,
                batch_size=stop - start,
                kv_cache_shape=tuple(chunk_cache.data.shape),
            )
            chunk_out, chunk_cache_data = reference.ref_kernel(
                (chunk_config, x[start:stop], chunk_cache)
            )
            torch.cuda.synchronize()
            output_chunks.append(chunk_out.cpu())
            token_chunks.append(
                chunk_cache_data[:, write_index : write_index + 1].cpu()
            )
            del chunk_cache, chunk_config, chunk_out, chunk_cache_data
            gc.collect()
            torch.cuda.empty_cache()
    del config, x, cache
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cat(output_chunks), torch.cat(token_chunks), write_index


def _run_checked(case: dict, custom_kernel, reference, torch):
    # Running candidate and reference side-by-side exceeds 80 GiB at prefill
    # 6144 because the reference materializes head-expanded K/V tensors.  Run
    # the reference first, retain only its small output and newly written cache
    # token, then regenerate the deterministic input for the candidate.  A
    # clone of the regenerated pristine cache is used to check that no other
    # cache locations changed.
    expected_out_cpu, expected_token_cpu, write_index = _run_reference_chunked(
        case, reference, torch
    )

    config, x, cache = reference.generate_input(**case)
    expected_cache = _clone_cache(cache, reference)
    expected_cache.data[:, write_index : write_index + 1].copy_(expected_token_cpu.cuda())
    expected_out = expected_out_cpu.cuda()
    with torch.no_grad():
        actual = custom_kernel((config, x, cache))
        torch.cuda.synchronize()
        max_error = _check_pair(actual, (expected_out, expected_cache.data), torch)
    del config, x, cache, expected_cache, actual, expected_out
    gc.collect()
    torch.cuda.empty_cache()
    return max_error


def evaluate_candidate(
    candidate: Path,
    official_root: Path,
    repeats: int,
    suite: str = "smoke",
    case_index: int | None = None,
) -> EvalSummary:
    import torch

    started = time.monotonic()
    if not torch.cuda.is_available():
        return EvalSummary(ok=False, error="CUDA is unavailable")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats()
    try:
        reference = _load_official_modules(official_root)
        custom_kernel = _load_candidate(candidate)
        if suite == "smoke":
            selected_cases = [("smoke", case) for case in SMOKE_CASES]
        elif suite == "full":
            selected_cases = [
                *[("correctness", case) for case in CORRECTNESS_CASES],
                *[("benchmark", case) for case in BENCHMARK_CASES],
            ]
        else:
            raise ValueError(f"unknown suite: {suite}")
        if case_index is not None:
            if not 0 <= case_index < len(selected_cases):
                raise ValueError(
                    f"case index {case_index} outside [0, {len(selected_cases)})"
                )
            selected_cases = [selected_cases[case_index]]

        passed = 0
        max_error = 0.0
        overall_peak_memory_gib = 0.0
        case_results: list[dict] = []
        score_means_us: list[float] = []
        for kind, base_case in selected_cases:
            case_started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            try:
                case_max_error = _run_checked(base_case, custom_kernel, reference, torch)
                durations: list[float] = []
                # Correctness above compiles and checks the shape before it is
                # timed.  Like the official benchmark path, timed calls are not
                # rechecked; input generation is excluded from the CUDA event.
                for repeat_index in range(max(3, repeats)):
                    case = dict(base_case)
                    case["seed"] += 13 * (repeat_index + 1)
                    config, x, cache = reference.generate_input(**case)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    with torch.no_grad():
                        start.record()
                        actual = custom_kernel((config, x, cache))
                        end.record()
                        torch.cuda.synchronize()
                        durations.append(float(start.elapsed_time(end) * 1000.0))
                    del config, x, cache, actual
                    gc.collect()
                    torch.cuda.empty_cache()
                mean_us = statistics.fmean(durations)
                passed += 1
                max_error = max(max_error, case_max_error)
                if suite == "smoke" or kind == "benchmark":
                    score_means_us.append(mean_us)
                case_results.append(
                    {
                        "kind": kind,
                        **base_case,
                        "ok": True,
                        "runs": len(durations),
                        "mean_us": mean_us,
                        "durations_us": durations,
                        "max_abs_error": case_max_error,
                        "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
                        "elapsed_s": time.monotonic() - case_started,
                        "error": "",
                    }
                )
            except Exception as exc:
                case_results.append(
                    {
                        "kind": kind,
                        **base_case,
                        "ok": False,
                        "runs": 0,
                        "mean_us": 0.0,
                        "durations_us": [],
                        "max_abs_error": 0.0,
                        "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
                        "elapsed_s": time.monotonic() - case_started,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                gc.collect()
                torch.cuda.empty_cache()
            overall_peak_memory_gib = max(
                overall_peak_memory_gib,
                torch.cuda.max_memory_allocated() / (1024**3),
            )

        return EvalSummary(
            ok=passed == len(selected_cases),
            score_us=geometric_mean(score_means_us) if score_means_us else 0.0,
            passed=passed,
            total=len(selected_cases),
            benchmark_means_us=score_means_us,
            case_results=case_results,
            max_abs_error=max_error,
            peak_memory_gib=overall_peak_memory_gib,
            elapsed_s=time.monotonic() - started,
            hardware=torch.cuda.get_device_name(0),
            suite=suite,
        )
    except Exception as exc:  # Candidate failures are evaluator outcomes.
        return EvalSummary(
            ok=False,
            passed=locals().get("passed", 0),
            total=len(locals().get("selected_cases", SMOKE_CASES)),
            max_abs_error=locals().get("max_error", 0.0),
            peak_memory_gib=torch.cuda.max_memory_allocated() / (1024**3),
            elapsed_s=time.monotonic() - started,
            hardware=torch.cuda.get_device_name(0),
            suite=suite,
            error=f"{type(exc).__name__}: {exc}",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--official-root",
        type=Path,
        default=DEFAULT_OFFICIAL_ROOT,
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = evaluate_candidate(
        args.candidate,
        args.official_root,
        args.repeats,
        suite=args.suite,
        case_index=args.case_index,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(summary), indent=2))
    print(json.dumps(asdict(summary)), flush=True)
    return 0 if summary.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
