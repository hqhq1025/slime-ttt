"""TTT-Discover MLA Decode environment adapted to one local NVIDIA A100."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from ..environment import RewardResult, TTTEnvironment
from ..state import State


class MLADecodeA100Env(TTTEnvironment):
    name = "mla_decode_a100"
    metric_name = "A100 MLA Decode smoke geometric-mean runtime (us)"
    entrypoint = "custom_kernel"
    maximize = False
    target = 3000.0

    def __init__(self, args):
        super().__init__(args)
        self.repeats = int(getattr(args, "ttt_mla_repeats", 3))
        self.gpu_device = str(getattr(args, "ttt_mla_gpu_device", "0"))
        self.official_root = Path(getattr(
            args,
            "ttt_mla_official_root",
            Path(__file__).resolve().parents[4].parent / "ttt-discover-official",
        ))
        self.warm_start = os.environ.get("TTT_MLA_WARM_START", "0") == "1"
        self.seed_code = ""
        # Measured by the checked-in A100 smoke evaluator (prefill 128/512,
        # three repeats each); callers can override after changing hardware.
        self.seed_score_us = float(os.environ.get("TTT_MLA_SEED_SCORE_US", "539.4624114971953"))
        if self.warm_start:
            self.seed_code = (
                self.official_root / "results/kernel-engineering/mla_code_3.py"
            ).read_text()

    def create_initial_states(self, n: int) -> list[State]:
        value = -self.seed_score_us if self.warm_start else -1_000_000.0
        return [State(timestep=-1, code=self.seed_code, construction=None, value=value) for _ in range(n)]

    def build_prompt(self, state: State) -> str:
        state_ctx = state.to_prompt(
            self.target, metric_name=self.metric_name, maximize=False, language="python"
        )
        return f"""You are optimizing the official TTT-Discover Multi-head Latent Attention (MLA) decode task.

Implement `custom_kernel(data)` where `data` is `(config, x, kv_cache)` and return `(output, updated_kv_cache)`.
The fixed DeepSeek-V3-derived configuration has batch size 128, decode length 1, hidden size 7168, 128 heads, q_lora_rank 1536, kv_lora_rank 512, qk_nope_head_dim 128, qk_rope_head_dim 64, and v_head_dim 128. Inputs and weights are bfloat16 CUDA tensors. Correctness must match the official reference at rtol=0.02 and atol=0.008, including the full KV cache.

This local evaluator is an NVIDIA A100 hardware-adaptation smoke: it uses the official prefill=128 and prefill=512 correctness shapes for correctness and local CUDA-event timing. It is not the official MI300X/H200 leaderboard and its score must not be compared to those published numbers.

{state_ctx}

Rules:
- Put all imports, Triton kernels, helpers, and `custom_kernel` in one final ```python block.
- Include a real `@triton.jit` kernel and preserve the exact official function contract.
- Do not use filesystem/network IO, subprocesses, hard-coded outputs, or gradients.
- Return only the final Python code block.
"""

    def evaluate(self, response_text: str, state: State) -> RewardResult:
        del state
        code = self.extract_code(response_text)
        if not code:
            return self.failure("no python code block found")
        if "@triton.jit" not in code:
            return self.failure("code must contain @triton.jit")

        worker = Path(__file__).resolve().parents[1] / "mla_decode_a100_eval.py"
        with tempfile.TemporaryDirectory(prefix="ttt_mla_decode_") as tmp_name:
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
                "--official-root",
                str(self.official_root),
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
                return self.failure(
                    f"MLA Decode evaluation timed out after {self.eval_timeout}s", exc.stdout or ""
                )
            stdout = proc.stdout[-8000:]
            if not output.exists():
                return self.failure(
                    f"MLA Decode evaluator exited {proc.returncode} without a result", stdout
                )
            try:
                result = json.loads(output.read_text())
            except Exception as exc:  # noqa: BLE001
                return self.failure(f"invalid evaluator result: {exc}", stdout)
            if not result.get("ok"):
                return self.failure(result.get("error", "MLA Decode evaluation failed"), stdout)

        score_us = float(result["score_us"])
        return RewardResult(
            reward=5000.0 / score_us,
            raw_score=score_us,
            correctness=1.0,
            construction=[],
            msg=(
                f"{self.metric_name}: {score_us:.3f}; "
                f"{result['passed']}/{result['total']} official smoke cases passed"
            ),
            stdout=stdout,
            metrics={
                "score_us": score_us,
                "tests_passed": int(result["passed"]),
                "max_abs_error": float(result["max_abs_error"]),
                "peak_memory_gib": float(result["peak_memory_gib"]),
                "eval_elapsed_s": float(result["elapsed_s"]),
            },
        )
