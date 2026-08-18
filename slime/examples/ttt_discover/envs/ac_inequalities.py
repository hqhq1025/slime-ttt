"""Autocorrelation inequality environments from the official TTT-Discover repo.

The two verifier formulas, initialization distribution, entrypoint names,
optimization directions, and prompt targets match ``examples/ac_inequalities``
at official commit 6c40e82.  The only integration change is execution through
slime's local subprocess sandbox rather than Tinker's evaluator.
"""

from __future__ import annotations

import inspect

import numpy as np

from ..environment import RewardResult, TTTEnvironment
from ..sandbox import run_python_entrypoint
from ..state import State


def evaluate_ac1(sequence: list[float]) -> float:
    if not isinstance(sequence, list) or not sequence:
        return np.inf
    if len(sequence) > 100000:
        return np.inf
    for x in sequence:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return np.inf
        if np.isnan(x) or np.isinf(x):
            return np.inf
    sequence = [min(1000.0, max(0.0, float(x))) for x in sequence]
    sum_a = np.sum(sequence)
    if sum_a < 0.01:
        return np.inf
    n = len(sequence)
    # The reference uses np.convolve, whose direct O(n^2) implementation can
    # monopolize the RolloutManager for tens of minutes at the documented
    # 100k-element limit. FFT convolution computes the same linear convolution
    # for long sequences and keeps verification bounded in practice.
    if n > 4096:
        from scipy.signal import fftconvolve

        convolution = fftconvolve(sequence, sequence, mode="full")
    else:
        convolution = np.convolve(sequence, sequence)
    return float(2 * n * np.max(convolution) / (sum_a**2))


def evaluate_ac2(sequence: list[float]) -> float:
    if not isinstance(sequence, list) or not sequence:
        raise ValueError("Invalid or empty sequence")
    if len(sequence) > 100000:
        raise ValueError("Sequence exceeds official 100000-element limit")
    for x in sequence:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError("Invalid sequence element type")
        if np.isnan(x) or np.isinf(x):
            raise ValueError("Invalid sequence element value")
    sequence = [min(1000.0, max(0.0, float(x))) for x in sequence]
    if np.sum(sequence) < 0.01:
        raise ValueError("Sum of sequence is too close to zero")

    if len(sequence) > 4096:
        from scipy.signal import fftconvolve

        convolution_2 = fftconvolve(sequence, sequence, mode="full")
    else:
        convolution_2 = np.convolve(sequence, sequence)
    x_points = np.linspace(-0.5, 0.5, len(convolution_2) + 2)
    x_intervals = np.diff(x_points)
    y_points = np.concatenate(([0], convolution_2, [0]))
    l2_norm_squared = 0.0
    for i in range(len(convolution_2) + 1):
        y1, y2 = y_points[i], y_points[i + 1]
        l2_norm_squared += (x_intervals[i] / 3) * (y1**2 + y1 * y2 + y2**2)
    norm_1 = np.sum(np.abs(convolution_2)) / (len(convolution_2) + 1)
    norm_inf = np.max(np.abs(convolution_2))
    return float(l2_norm_squared / (norm_1 * norm_inf))


class _AutoCorrBase(TTTEnvironment):
    max_construction_len = 100000
    evaluator = None

    def create_initial_states(self, n: int) -> list[State]:
        # Official code reinitializes this seed for each initial state.
        states = []
        for _ in range(n):
            rng = np.random.default_rng(12345)
            construction = [float(rng.random())] * int(rng.integers(1000, 8000))
            score = self.evaluator(construction)
            states.append(State(
                timestep=-1,
                construction=construction,
                value=self.value_from_score(score),
            ))
        return states

    def evaluate(self, response_text: str, state: State) -> RewardResult:
        code = self.extract_code(response_text)
        if not code:
            return self.failure("no python code block found")
        helper_name = "evaluate_sequence"
        preamble = "import numpy as np\n" + inspect.getsource(self.evaluator)
        preamble += f"\n{helper_name} = {self.evaluator.__name__}\n"
        if state.construction is not None:
            preamble += f"height_sequence_1 = {list(state.construction)!r}\n"
        res = run_python_entrypoint(
            code, self.entrypoint, self.eval_timeout, preamble=preamble,
            cpus=list(range(self.num_cpus_per_task)),
        )
        if not res.ok:
            return self.failure(res.error, stdout=res.stdout)
        try:
            construction = list(res.value)
            score = float(self.evaluator(construction))
            if not np.isfinite(score):
                raise ValueError("non-finite score")
        except Exception as exc:  # noqa: BLE001
            return self.failure(f"verification failed: {exc}", stdout=res.stdout)
        reward = score if self.maximize else 1.0 / (1e-8 + score)
        return RewardResult(
            reward=float(reward), raw_score=score, correctness=1.0,
            construction=construction, msg=f"{self.metric_name}: {score:.8f}",
            stdout=res.stdout, metrics={self.metric_name.replace(" ", "_"): score,
                                        "sequence_length": len(construction)},
        )

    def build_prompt(self, state: State) -> str:
        direction = "maximizes" if self.maximize else "minimizes"
        state_ctx = state.to_prompt(
            self.target, metric_name=self.metric_name, maximize=self.maximize,
        )
        return f"""Act as an expert software developer and inequality specialist.

Generate a non-negative step-function height sequence that {direction} the
autocorrelation quantity described below. The injected `evaluate_sequence`
function is the exact official verifier and may be called as often as needed.

{self.formula}

{state_ctx}
Length of the current construction: {len(state.construction or [])}.
It is available as the global `height_sequence_1`. Explore improved algorithms,
initializations, and hyperparameters rather than merely returning it unchanged.

Rules:
- Define `{self.entrypoint}(seed=42, budget_s={self.eval_timeout}, **kwargs)` and return a Python list.
- Values must be finite and non-negative; length must be between 1 and 100000.
- numpy, scipy, cvxpy, and math are available. Use at most {self.num_cpus_per_task} CPUs.
- Helpers must be top-level; no filesystem or network IO.
- Return the final program inside a ```python code block.
"""


class AC1Env(_AutoCorrBase):
    name = "autocorrelation_ac1"
    metric_name = "AC1 upper bound"
    maximize = False
    target = 1.5030
    entrypoint = "propose_candidate"
    evaluator = staticmethod(evaluate_ac1)
    formula = (
        "For a sequence f of length n, minimize "
        "2*n*max(convolve(f,f))/(sum(f)**2). Lower is better."
    )


class AC2Env(_AutoCorrBase):
    name = "autocorrelation_ac2"
    metric_name = "AC2 lower bound"
    maximize = True
    target = 0.97
    entrypoint = "construct_function"
    evaluator = staticmethod(evaluate_ac2)
    formula = (
        "For c=convolve(f,f), maximize ||c||_2^2/(||c||_1*||c||_inf), "
        "using the injected official piecewise-linear integration verifier."
    )
