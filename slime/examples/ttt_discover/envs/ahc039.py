"""AHC039 test-time discovery environment using official released artifacts."""

from __future__ import annotations

import os
import re
from pathlib import Path

from ..ahc_judge import judge_ahc_public
from ..environment import RewardResult, TTTEnvironment
from ..state import State


def _load_official_artifacts(reference_root: str | Path) -> tuple[str, str, float]:
    prompt_py = Path(reference_root) / "examples" / "ahc" / "prompt.py"
    if not prompt_py.is_file():
        raise FileNotFoundError(f"official AHC prompt artifact not found: {prompt_py}")
    # Read the triple-quoted payloads verbatim. Importing prompt.py would let
    # Python interpret C++ escapes such as "\n", corrupting the released seed
    # into an uncompilable program with a literal newline inside the C++ string.
    source = prompt_py.read_text()

    def payload(name: str) -> str:
        marker = f"{name} = '''"
        start = source.find(marker)
        if start < 0:
            raise ValueError(f"{name} missing from {prompt_py}")
        start += len(marker)
        end = source.find("'''", start)
        if end < 0:
            raise ValueError(f"unterminated {name} in {prompt_py}")
        return source[start:end]

    value_match = re.search(r"AHC039_BEST_CODE_VALUE\s*=\s*([0-9.]+)", source)
    if value_match is None:
        raise ValueError(f"AHC039_BEST_CODE_VALUE missing from {prompt_py}")
    return payload("AHC039_PROMPT"), payload("AHC039_BEST_CODE"), float(value_match.group(1))


class AHC039Env(TTTEnvironment):
    name = "ahc039"
    metric_name = "mean public score"
    code_language = "cpp"
    maximize = True
    target = 5000.0

    def __init__(self, args):
        super().__init__(args)
        self.cache_dir = Path(
            os.environ.get("AHC_CACHE", getattr(args, "ttt_ahc_cache_dir"))
        )
        self.reference_root = Path(
            os.environ.get(
                "TTT_OFFICIAL_ROOT", getattr(args, "ttt_ahc_reference_root")
            )
        )
        self.max_cases = int(getattr(args, "ttt_ahc_max_cases", 1))
        self.time_limit = float(getattr(args, "ttt_ahc_time_limit", 2.0))
        self.reward_scale = float(getattr(args, "ttt_ahc_reward_scale", 1500.0))
        self.problem_prompt, self.seed_code, self.seed_value = _load_official_artifacts(
            self.reference_root
        )

    def create_initial_states(self, n: int) -> list[State]:
        return [
            State(
                timestep=-1,
                code=self.seed_code,
                value=self.seed_value,
                construction=None,
            )
            for _ in range(n)
        ]

    def build_prompt(self, state: State) -> str:
        previous = state.to_prompt(
            self.target, metric_name="performance", maximize=True, language="cpp"
        )
        return f"""{self.problem_prompt}

{previous}

Rules:
- You must use C++20 to solve the problem.
- Define all code in one final ```cpp``` block.
- In the final response, output only the program and no explanatory text.
- Try diverse approaches and respect the 2 second official time limit.
"""

    def evaluate(self, response_text: str, state: State) -> RewardResult:
        del state
        code = self.extract_code(response_text)
        if not code:
            return self.failure("no cpp code block found")
        result = judge_ahc_public(
            code,
            problem_id="ahc039",
            cache_dir=self.cache_dir,
            max_cases=self.max_cases,
            time_limit=self.time_limit,
        )
        reward = result.raw_score / self.reward_scale if result.status == "ok" else 0.0
        return RewardResult(
            reward=float(reward),
            raw_score=float(result.raw_score),
            correctness=1.0 if result.status == "ok" else 0.0,
            construction=None,
            msg=f"{result.status}: {result.msg}"[:300],
            metrics={
                "judge_status": result.status,
                "passed_cases": result.passed,
                "num_cases": result.num_cases,
                "compile_seconds": result.compile_seconds,
                "run_seconds": result.run_seconds,
                "public_only": True,
            },
        )
