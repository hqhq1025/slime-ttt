"""Environment interface for TTT-Discover-on-slime.

An environment defines one open problem: how to seed it (``create_initial_states``),
how to turn a discovery state into a prompt (``build_prompt``), and how to score a
model response (``evaluate``). This mirrors the reference ``Environment`` /
``SandboxRewardEvaluator`` split but is reduced to exactly what the slime rollout
needs — there is no Tinker renderer, no per-step MDP machinery (TTT episodes are
single-step), and reward evaluation goes through :mod:`examples.ttt_discover.sandbox`.

To add a domain, subclass :class:`TTTEnvironment` and point ``--ttt-env-path`` at
your subclass (e.g. ``examples.ttt_discover.envs.erdos.ErdosMinOverlapEnv``).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .state import State


@dataclass
class RewardResult:
    """Outcome of scoring one response.

    ``reward`` is the scalar the RL objective optimizes (the entropic estimator
    centers it within the group). ``raw_score`` is the human-facing metric.
    ``construction`` is the object carried forward to seed future prompts.
    """

    reward: float
    raw_score: float
    correctness: float
    construction: Any = None
    msg: str = ""
    stdout: str = ""
    metrics: dict = field(default_factory=dict)


def extract_last_code_block(text: str, language: str = "python") -> str | None:
    """Return the last code block, accepting an EOF-truncated closing fence.

    The official TTT-Discover parser intentionally accepts a code block that
    reaches the end of the response. Long reasoning rollouts frequently hit the
    generation limit after producing useful code but before the final fence.
    """
    pattern = re.compile(
        rf"```{re.escape(language)}\s*\n(.*?)(?:\n```|```)?(?=\n```|$)",
        re.DOTALL,
    )
    matches = pattern.findall(text)
    if not matches:
        return None
    code = matches[-1].strip()
    return code or None


class TTTEnvironment(ABC):
    # Identity / display
    name: str = "ttt"
    metric_name: str = "value"
    code_language: str = "python"
    entrypoint: str = "run"  # function the sandbox calls in the generated program

    # Optimization direction of ``raw_score`` (True = larger is better). The
    # internal ``State.value`` always stores "higher is better", so minimization
    # envs negate the score.
    maximize: bool = True
    target: float = 0.0  # shown in the prompt as the goal

    def __init__(self, args):
        self.args = args
        self.eval_timeout = int(getattr(args, "ttt_eval_timeout", 600))
        self.num_cpus_per_task = int(getattr(args, "ttt_num_cpus_per_task", 1))

    # ---- to implement -------------------------------------------------
    @abstractmethod
    def create_initial_states(self, n: int) -> list[State]:
        """Return ``n`` seed states (random restarts) to populate the archive."""

    @abstractmethod
    def build_prompt(self, state: State) -> str:
        """Build the user prompt, conditioned on ``state`` (best-so-far)."""

    @abstractmethod
    def evaluate(self, response_text: str, state: State) -> RewardResult:
        """Parse + sandbox-run the response and score it against ``state``."""

    # ---- helpers ------------------------------------------------------
    def extract_code(self, response_text: str) -> str | None:
        return extract_last_code_block(response_text, self.code_language)

    def value_from_score(self, raw_score: float) -> float:
        """Map a domain raw score to the archive's "higher is better" value."""
        return float(raw_score) if self.maximize else -float(raw_score)

    def make_child_state(self, step: int, code: str, result: RewardResult) -> State:
        return State(
            timestep=step,
            code=code,
            construction=result.construction,
            value=self.value_from_score(result.raw_score),
            observation=result.stdout or "",
        )

    def failure(self, msg: str, stdout: str = "") -> RewardResult:
        return RewardResult(
            reward=float(getattr(self.args, "ttt_fail_reward", 0.0)),
            raw_score=0.0,
            correctness=0.0,
            construction=None,
            msg=msg,
            stdout=stdout,
        )
