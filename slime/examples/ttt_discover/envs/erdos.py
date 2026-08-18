"""Erdős minimum-overlap environment (faithful port of the reference example).

Goal: find a step function ``h: [0,2] -> [0,1]`` with ``∫h = 1`` that minimizes
the overlap ``C5 = max_k ∫ h(x)(1 - h(x+k)) dx``. Lower is better; the human
record is ~0.380927 and TTT-Discover reached 0.380876.

This shows the full contract: random seeding, prompt conditioning on the
best-so-far construction (exposed to the generated program as ``initial_h_values``),
and a sandbox-run reward of ``1 / c5_bound`` with ``raw_score = c5_bound``.
"""

from __future__ import annotations

import inspect

import numpy as np

from ..environment import RewardResult, TTTEnvironment
from ..sandbox import run_python_entrypoint
from ..state import State


def verify_c5_solution(h_values, c5_achieved, n_points):
    """Validate a candidate and recompute its C5. Injected into the sandbox."""
    import numpy as np

    h_values = np.asarray(h_values, dtype=np.float64)
    if h_values.shape != (n_points,):
        raise ValueError(f"Expected h shape ({n_points},), got {h_values.shape}")
    if not np.all(np.isfinite(h_values)):
        raise ValueError("h_values contain NaN or inf")
    if np.any(h_values < 0) or np.any(h_values > 1):
        raise ValueError("h(x) not in [0, 1]")
    target_sum = n_points / 2.0
    current_sum = float(np.sum(h_values))
    if current_sum != target_sum:
        h_values = h_values * (target_sum / current_sum)
        if np.any(h_values < 0) or np.any(h_values > 1):
            raise ValueError("after normalization h(x) not in [0, 1]")
    dx = 2.0 / n_points
    correlation = np.correlate(h_values, 1.0 - h_values, mode="full") * dx
    computed = float(np.max(correlation))
    if not np.isfinite(computed):
        raise ValueError("computed C5 not finite")
    if not np.isclose(computed, c5_achieved, atol=1e-4):
        raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed:.6f}")
    return computed


class ErdosMinOverlapEnv(TTTEnvironment):
    name = "erdos_min_overlap"
    metric_name = "C5 bound"
    maximize = False  # minimize the bound
    target = 0.3808
    entrypoint = "run"
    max_construction_len = 1000

    def create_initial_states(self, n: int) -> list[State]:
        rng = np.random.default_rng()
        states = []
        for _ in range(n):
            n_points = int(rng.integers(40, 100))
            constr = np.ones(n_points) * 0.5 + (rng.uniform(-0.4, 0.4, n_points) - 0.0)
            constr = constr - np.mean(constr) + 0.5  # keep mean ~ 0.5 (∫h ~ 1)
            dx = 2.0 / n_points
            c5 = float(np.max(np.correlate(constr, 1 - constr, mode="full") * dx))
            states.append(State(timestep=-1, code="", construction=constr.tolist(),
                                value=-c5))  # higher-is-better => negate
        return states

    def build_prompt(self, state: State) -> str:
        state_ctx = state.to_prompt(self.target, metric_name="C₅ bound", maximize=False)
        constr = state.construction
        construction_section = ""
        if constr is not None and len(constr) > 0:
            construction_section = (
                f"\nYou may start from the current construction, available as the global "
                f"`initial_h_values` (n={len(constr)} samples). You are encouraged to also try "
                f"other starting points to escape local optima.\n"
            )
        if state.code and state.code.strip():
            code_section = (
                "Reason about how to further improve this construction. Try something different "
                "from the above (new algorithmic ideas, different heuristics, swept hyperparameters). "
                "Unless you make a meaningful improvement, you will not be rewarded."
            )
        else:
            code_section = "Write code to optimize this construction."

        return f"""You are an expert in harmonic analysis, numerical optimization, and mathematical discovery.
Your task is to find an improved upper bound for the Erdős minimum overlap constant C₅.

## Problem
Find a step function h: [0, 2] → [0, 1] that **minimizes** the overlap integral:

    C₅ = max_k ∫ h(x)(1 - h(x+k)) dx

Constraints: 0 ≤ h(x) ≤ 1 and ∫₀² h(x) dx = 1.
Discretize h as n_points samples with dx = 2/n_points; then sum(h)*dx = 1 (i.e. sum(h) == n_points/2).
Evaluation computes C₅ = max(np.correlate(h, 1-h, mode="full") * dx). **Lower is better.**

## Rules
- Define `run(seed=42, budget_s={self.eval_timeout}, **kwargs)` returning `(h_values, c5_bound, n_points)`.
- Use numpy/scipy/math. Top-level helper functions only (no closures/lambdas for heavy work). No file or network IO.
- `initial_h_values` (if available) is pre-injected. Keep n_points < 1000 (faster to evaluate).
- Finish within the time budget and return the best solution found.

Current record: C₅ ≤ 0.38092. Goal: show C₅ ≤ 0.38080.

{state_ctx}
{construction_section}
{code_section}
"""

    def evaluate(self, response_text: str, state: State) -> RewardResult:
        code = self.extract_code(response_text)
        if not code:
            return self.failure("no python code block found")

        # Preamble: the verifier (so the program can self-check) plus the
        # best-so-far construction exposed as ``initial_h_values``.
        preamble = "import numpy as np\n\n" + inspect.getsource(verify_c5_solution) + "\n"
        if state.construction is not None:
            preamble += f"initial_h_values = np.array({list(state.construction)!r})\n"

        res = run_python_entrypoint(
            code, self.entrypoint, self.eval_timeout, preamble=preamble,
            cpus=list(range(self.num_cpus_per_task)),
        )
        if not res.ok:
            return self.failure(res.error, stdout=res.stdout)

        try:
            h_values, c5_bound, n_points = res.value
            c5_bound = verify_c5_solution(h_values, c5_bound, n_points)
            if not np.isfinite(c5_bound) or c5_bound <= 0:
                return self.failure("invalid C5", stdout=res.stdout)
        except Exception as e:  # noqa: BLE001 - any malformed return is a failure
            return self.failure(f"verification failed: {e}", stdout=res.stdout)

        return RewardResult(
            reward=float(1.0 / (1e-8 + c5_bound)),  # lower bound => higher reward
            raw_score=float(c5_bound),
            correctness=1.0,
            construction=list(np.asarray(h_values, dtype=np.float64)),
            msg=f"C5 bound: {c5_bound:.6f}",
            stdout=res.stdout,
            metrics={"c5_bound": float(c5_bound), "n_points": int(n_points)},
        )
