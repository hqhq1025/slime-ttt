"""Native public-case judge for the AHC tasks released with TTT-Discover.

The official artifact evaluates with ALE-Bench.  Its released cache contains
the 150 public inputs and a problem-specific ``*_tester`` binary.  This module
uses those exact inputs/testers but runs the candidate directly with the host
``g++`` instead of Docker.  It is therefore suitable for local smoke/training,
but it is not a private AtCoder submission score.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path


_SCORE_RE = re.compile(r"Score\s*=\s*(-?\d+)")
_CPP_FLAGS = (
    "-std=gnu++20",
    "-O2",
    "-DONLINE_JUDGE",
    "-DATCODER",
    "-Wall",
    "-Wextra",
    "-mtune=native",
    "-march=native",
)


@dataclass
class AHCJudgeResult:
    raw_score: float
    status: str
    passed: int
    num_cases: int
    case_scores: list[int] = field(default_factory=list)
    msg: str = ""
    compile_seconds: float = 0.0
    run_seconds: float = 0.0


def public_inputs(cache_dir: str | Path, problem_id: str) -> list[Path]:
    root = Path(cache_dir)
    inputs = root / "public_inputs_150" / f"{problem_id}_inputs"
    return sorted(inputs.glob(f"{problem_id}_*_input.txt"))


def validate_cache(cache_dir: str | Path, problem_id: str) -> tuple[Path, list[Path]]:
    root = Path(cache_dir)
    tester = root / "tester_binaries" / f"{problem_id}_tester"
    inputs = public_inputs(root, problem_id)
    if not tester.is_file():
        raise FileNotFoundError(f"AHC tester not found: {tester}")
    if not inputs:
        raise FileNotFoundError(
            f"AHC public inputs not found under {root / 'public_inputs_150'}"
        )
    return tester, inputs


def _run(command: list[str], *, stdin=None, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def judge_ahc_public(
    code: str,
    *,
    problem_id: str,
    cache_dir: str | Path,
    max_cases: int = 1,
    time_limit: float = 2.0,
    compile_timeout: float = 90.0,
) -> AHCJudgeResult:
    """Compile and score ``code`` on released public cases.

    ALE-Bench reports an entire submission as invalid when any case fails.  We
    mirror that behavior by returning raw score zero unless every selected case
    compiles, runs, and is accepted.  On success ``raw_score`` is the arithmetic
    mean used by the official TTT AHC reward evaluator.
    """
    try:
        tester, inputs = validate_cache(cache_dir, problem_id)
    except (FileNotFoundError, OSError) as exc:
        return AHCJudgeResult(0.0, "cache_error", 0, 0, msg=str(exc))
    if max_cases > 0:
        inputs = inputs[:max_cases]

    with tempfile.TemporaryDirectory(prefix=f"ttt_{problem_id}_") as tmp_name:
        tmp = Path(tmp_name)
        source = tmp / "Main.cpp"
        executable = tmp / "a.out"
        source.write_text(code)

        t0 = time.monotonic()
        try:
            compiled = _run(
                ["g++", *_CPP_FLAGS, str(source), "-o", str(executable)],
                timeout=compile_timeout,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return AHCJudgeResult(
                0.0, "compile_error", 0, len(inputs), msg=str(exc),
                compile_seconds=time.monotonic() - t0,
            )
        compile_seconds = time.monotonic() - t0
        if compiled.returncode != 0:
            msg = compiled.stderr.decode("utf-8", "replace")[-2000:]
            return AHCJudgeResult(
                0.0, "compile_error", 0, len(inputs), msg=msg,
                compile_seconds=compile_seconds,
            )

        scores: list[int] = []
        run_seconds = 0.0
        for input_path in inputs:
            output_path = tmp / f"{input_path.stem}.out"
            t1 = time.monotonic()
            try:
                with input_path.open("rb") as stdin:
                    candidate = _run(
                        [str(executable)], stdin=stdin, timeout=time_limit + 1.0
                    )
            except subprocess.TimeoutExpired:
                return AHCJudgeResult(
                    0.0, "tle", len(scores), len(inputs), case_scores=scores,
                    msg=f"candidate exceeded {time_limit + 1.0:.1f}s on {input_path.name}",
                    compile_seconds=compile_seconds,
                    run_seconds=run_seconds + time.monotonic() - t1,
                )
            except OSError as exc:
                return AHCJudgeResult(
                    0.0, "runtime_error", len(scores), len(inputs), case_scores=scores,
                    msg=str(exc), compile_seconds=compile_seconds,
                    run_seconds=run_seconds + time.monotonic() - t1,
                )
            run_seconds += time.monotonic() - t1
            if candidate.returncode != 0:
                return AHCJudgeResult(
                    0.0, "runtime_error", len(scores), len(inputs), case_scores=scores,
                    msg=candidate.stderr.decode("utf-8", "replace")[-1000:],
                    compile_seconds=compile_seconds, run_seconds=run_seconds,
                )
            output_path.write_bytes(candidate.stdout)

            checked = _run(
                [str(tester), str(input_path), str(output_path)], timeout=30.0
            )
            judge_text = (checked.stdout + checked.stderr).decode("utf-8", "replace")
            match = _SCORE_RE.search(judge_text)
            if checked.returncode != 0 or match is None:
                return AHCJudgeResult(
                    0.0, "wa", len(scores), len(inputs), case_scores=scores,
                    msg=judge_text[-1000:], compile_seconds=compile_seconds,
                    run_seconds=run_seconds,
                )
            scores.append(int(match.group(1)))

    return AHCJudgeResult(
        raw_score=sum(scores) / len(scores) if scores else 0.0,
        status="ok",
        passed=len(scores),
        num_cases=len(inputs),
        case_scores=scores,
        msg=f"{len(scores)}/{len(inputs)} released public cases",
        compile_seconds=compile_seconds,
        run_seconds=run_seconds,
    )
