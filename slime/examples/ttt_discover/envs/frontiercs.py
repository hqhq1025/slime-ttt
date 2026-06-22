"""Frontier-CS environment: test-time discovery of C++ solutions to one problem.

Frontier-CS problems are competitive-programming / optimization tasks graded by a
points-based testlib checker. We apply the TTT-Discover loop to a **single**
target problem (``--ttt-frontiercs-problem-id``): the model proposes C++
solutions, the judge returns a score, the entropic objective pushes toward the
best ones, and the archive feeds the best-so-far solution back into the prompt.

Reward = judge points (higher is better). The judge has two backends
(``--ttt-judge-backend``):
  - ``remote`` : POST to the user's judge server (full/private tests, real score)
  - ``local``  : compile + run public testdata + testlib checker (smoke only)
See ``frontiercs_judge.py``. The remote protocol is the one hook to finalize once
the user describes the judge.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..environment import RewardResult, TTTEnvironment
from ..frontiercs_judge import score_solution
from ..state import State

_SYSTEM_MSG = (
    "You are an expert competitive programmer. You will be given a problem "
    "specification and must produce a correct, efficient C++ solution that passes "
    "the tests and maximizes the score."
)
_FORMAT_MSG = (
    "Read input from stdin, solve the problem, and write the answer to stdout. "
    "Output only the program, enclosed in a single ```cpp code block."
)


def _extract_cpp(text: str) -> str | None:
    m = re.search(r"```(?:cpp|c\+\+|cxx)?\s*\n(.*?)```", text, re.DOTALL)
    if not m:
        return None
    code = m.group(1).strip()
    return code or None


class FrontierCSEnv(TTTEnvironment):
    name = "frontiercs"
    metric_name = "score"
    code_language = "cpp"
    maximize = True  # judge points: higher is better

    def __init__(self, args):
        super().__init__(args)
        self.problems_dir = getattr(args, "ttt_frontiercs_problems_dir", None)
        self.problem_id = str(getattr(args, "ttt_frontiercs_problem_id", "0"))
        self.judge_backend = getattr(args, "ttt_judge_backend", "local")
        self.judge_url = getattr(args, "ttt_judge_url", None)
        self.judge_max_cases = int(getattr(args, "ttt_judge_max_cases", 1))
        self.judge_score_key = getattr(args, "ttt_judge_score_key", "score")
        self.judge_poll_interval = float(getattr(args, "ttt_judge_poll_interval", 3.0))
        self.max_score = float(getattr(args, "ttt_frontiercs_max_score", 100.0))
        self.target = self.max_score
        # Reward mode: "score" (original normalized points) or "sum_value"/"mean_value"
        # (dense per-case objective Value — nonzero even when points are 0, e.g. 159).
        self.reward_mode = getattr(args, "ttt_frontiercs_reward_mode", "score")
        self.value_scale = float(getattr(args, "ttt_frontiercs_value_scale", 1.0))
        if not self.problems_dir:
            raise ValueError("FrontierCSEnv requires --ttt-frontiercs-problems-dir")
        pdir = Path(self.problems_dir) / self.problem_id
        stmt = pdir / "statement.txt"
        if not stmt.exists():
            raise FileNotFoundError(f"statement not found: {stmt}")
        self.statement = stmt.read_text()

    def create_initial_states(self, n: int) -> list[State]:
        # Seed with empty solutions; the model writes the first program from scratch.
        return [State(timestep=-1, code="", construction=None, value=None) for _ in range(n)]

    def extract_code(self, response_text: str) -> str | None:
        return _extract_cpp(response_text)

    def build_prompt(self, state: State) -> str:
        base = (
            f"{_SYSTEM_MSG}\n\n{self.statement}\n\n"
            f"### Format: {_FORMAT_MSG}\n"
        )
        if state.code and state.code.strip() and state.value is not None:
            if self.reward_mode == "sum_value":
                metric, tgt, goal = "the total objective Value (summed over all test cases)", None, "a higher total Value"
            elif self.reward_mode == "mean_value":
                metric, tgt, goal = "the mean objective Value per test case", None, "a higher Value"
            else:  # "score" (original)
                metric, tgt, goal = "score", self.target, "a higher score"
            cond = state.to_prompt(tgt, metric_name=metric, maximize=True, language="cpp")
            base += (
                f"\nYou have already produced a solution; improve it to achieve {goal} "
                "(fix failures, handle more cases, optimize the objective). Try a "
                "meaningfully different approach if you are stuck.\n"
                f"{cond}\n"
            )
        base += "\n```cpp\n// YOUR SOLUTION HERE\n```\n### Answer:\n"
        return base

    def evaluate(self, response_text: str, state: State) -> RewardResult:
        code = self.extract_code(response_text)
        if not code:
            return self.failure("no cpp code block found")
        res = score_solution(
            self.problem_id,
            code,
            backend=self.judge_backend,
            problems_dir=self.problems_dir,
            judge_url=self.judge_url,
            timeout=float(getattr(self.args, "ttt_judge_timeout", 300.0)),
            max_cases=self.judge_max_cases,
            score_key=self.judge_score_key,
            poll_interval=self.judge_poll_interval,
            max_score=self.max_score,
        )
        # "ran" = produced a real run verdict (so the code is worth keeping in the
        # archive to iterate on). Compile errors / transport errors / timeouts are not.
        ran = ("compile" not in res.status.lower()) and res.status not in ("error", "timeout")
        # Reward depends on --ttt-frontiercs-reward-mode:
        #   score      -> normalized judge points in [0,1] (original behavior)
        #   sum_value  -> sum of per-case objective Value (dense; nonzero at 0 points)
        #   mean_value -> sum_value / num_cases
        # construction = the code so the next prompt is seeded with this candidate.
        if self.reward_mode == "sum_value":
            raw = res.sum_value
            reward = raw * self.value_scale
        elif self.reward_mode == "mean_value":
            raw = res.mean_value
            reward = raw * self.value_scale
        else:  # "score" (default, unchanged)
            raw = res.score
            reward = res.normalized
        return RewardResult(
            reward=float(reward),
            raw_score=float(raw),
            correctness=1.0 if ran else 0.0,
            construction=code,
            msg=f"{res.status}: {res.msg}"[:200],
            stdout="",
            metrics={"judge_status": res.status, "judge_score": float(res.score),
                     "score_unbounded": (res.detail or {}).get("scoreUnbounded"),
                     "sum_value": float(res.sum_value), "num_cases": res.num_cases},
        )
