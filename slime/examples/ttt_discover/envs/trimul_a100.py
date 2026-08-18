"""TTT-Discover environment for the official TriMul task on local A100s."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from ..environment import RewardResult, TTTEnvironment
from ..state import State


class TriMulA100Env(TTTEnvironment):
    name = "trimul_a100"
    metric_name = "TriMul geometric-mean runtime (us)"
    entrypoint = "custom_kernel"
    maximize = False
    target = 1000.0

    def __init__(self, args):
        super().__init__(args)
        self.profile = str(getattr(args, "ttt_trimul_profile", "smoke"))
        self.repeats = int(getattr(args, "ttt_trimul_repeats", 5))
        self.gpu_device = str(getattr(args, "ttt_trimul_gpu_device", "0"))
        self.warm_start = os.environ.get("TTT_TRIMUL_WARM_START", "0") == "1"
        self.seed_code = ""
        self.seed_score_us = float(os.environ.get("TTT_TRIMUL_SEED_SCORE_US", "903.62"))
        if self.warm_start:
            official_root = Path(os.environ.get(
                "TTT_OFFICIAL_ROOT",
                Path(__file__).resolve().parents[4].parent / "ttt-discover-official",
            ))
            released = official_root / "results/kernel-engineering/trimul.py"
            self.seed_code = released.read_text().replace(
                'cfg.get("nomask", True)', 'cfg.get("nomask", False)'
            )

    def create_initial_states(self, n: int) -> list[State]:
        # Matches the official GPU-mode environment: no baseline program is
        # assumed unless the local weak-model smoke explicitly enables the
        # released-kernel warm start.
        value = -self.seed_score_us if self.warm_start else -1_000_000.0
        return [
            State(timestep=-1, code=self.seed_code, construction=None, value=value)
            for _ in range(n)
        ]

    def build_prompt(self, state: State) -> str:
        state_ctx = state.to_prompt(
            self.target, metric_name=self.metric_name, maximize=False, language="python"
        )
        scope = (
            "the complete official 18-test/7-benchmark suite"
            if self.profile == "full"
            else "a smoke subset of 4 official tests and 2 official leaderboard benchmarks"
        )
        return f"""You are an expert Triton engineer optimizing the outgoing Triangle Multiplicative Update used by AlphaFold-style models.

Implement `custom_kernel(data)` where `data` is `(input, mask, weights, config)`:
- input: float32 CUDA tensor `[B, N, N, C]`
- mask: CUDA tensor `[B, N, N]`, either all ones or random zeros/ones
- config contains `dim` and `hidden_dim`
- output: float32 CUDA tensor `[B, N, N, C]`

The exact reference is:
1. LayerNorm over C.
2. Five bias-free projections: left/right, left/right sigmoid gates, output gate.
3. Apply mask and gates to left/right.
4. `einsum('...ikd,...jkd->...ijd', left, right)`.
5. LayerNorm over hidden dim, multiply output gate, final linear projection.

Constraints follow the official TriMul contest: B in {{1,2}}, N up to 1024, C in {{128,256,384,768}}, hidden_dim=128, normal or Cauchy inputs. Correctness uses rtol=atol=0.02. Ranking is the geometric mean of CUDA-event mean runtimes; lower is better. This local run evaluates {scope} on an NVIDIA A100 with installed Triton. The actual `mask` tensor must always be honored; do not rely on a `config['nomask']` field.

{state_ctx}

Rules:
- Define all imports, Triton kernels, helpers, and `custom_kernel` in one final ```python block.
- At least one operation must use an `@triton.jit` kernel.
- Do not use identity shortcuts, filesystem/network IO, subprocesses, or gradients.
- Return only the output tensor from `custom_kernel`.
"""

    def evaluate(self, response_text: str, state: State) -> RewardResult:
        del state
        code = self.extract_code(response_text)
        if not code:
            return self.failure("no python code block found")
        if "@triton.jit" not in code:
            return self.failure("code must contain @triton.jit")
        if "identity" in code.lower():
            return self.failure("identity kernels are not allowed")

        worker = Path(__file__).resolve().parents[1] / "trimul_eval.py"
        with tempfile.TemporaryDirectory(prefix="ttt_trimul_") as tmp_name:
            tmp = Path(tmp_name)
            candidate = tmp / "candidate.py"
            output = tmp / "result.json"
            candidate.write_text(code)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = self.gpu_device
            cmd = [
                sys.executable,
                str(worker),
                str(candidate),
                "--profile",
                self.profile,
                "--repeats",
                str(self.repeats),
                "--output",
                str(output),
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=self.eval_timeout,
                    env=env,
                    start_new_session=True,
                )
            except subprocess.TimeoutExpired as exc:
                return self.failure(f"TriMul evaluation timed out after {self.eval_timeout}s", exc.stdout or "")
            stdout = proc.stdout[-8000:]
            if not output.exists():
                return self.failure(f"TriMul evaluator exited {proc.returncode} without a result", stdout)
            try:
                result = json.loads(output.read_text())
            except Exception as exc:  # noqa: BLE001
                return self.failure(f"invalid evaluator result: {exc}", stdout)
            if not result.get("ok"):
                return self.failure(result.get("error", "TriMul evaluation failed"), stdout)

        score_us = float(result["score_us"])
        reward = 1500.0 / score_us
        return RewardResult(
            reward=reward,
            raw_score=score_us,
            correctness=1.0,
            construction=[],
            msg=f"{self.metric_name}: {score_us:.3f}; {result['passed']}/{result['total']} tests passed",
            stdout=stdout,
            metrics={
                "score_us": score_us,
                "tests_passed": int(result["passed"]),
                "max_abs_error": float(result["max_abs_error"]),
                "eval_elapsed_s": float(result["elapsed_s"]),
            },
        )
