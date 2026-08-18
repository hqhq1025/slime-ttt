"""Official TTT-Discover circle-packing verifier for n=26 and n=32."""

from __future__ import annotations

import inspect

import numpy as np

from ..environment import RewardResult, TTTEnvironment
from ..sandbox import run_python_entrypoint
from ..state import State


def validate_packing(centers, radii):
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    n = centers.shape[0]
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if r < 0 or x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            if np.sqrt(np.sum((centers[i] - centers[j]) ** 2)) < radii[i] + radii[j] - 1e-12:
                return False
    return True


class _CirclePackingBase(TTTEnvironment):
    name = "circle_packing"
    metric_name = "sum of radii"
    maximize = True
    entrypoint = "run_packing"
    num_circles = 26

    def create_initial_states(self, n: int) -> list[State]:
        # The official environment starts from an empty State and does not reuse
        # the returned construction; code and score are still retained by slime.
        return [State(timestep=-1, construction=[], value=0.0) for _ in range(n)]

    def build_prompt(self, state: State) -> str:
        state_ctx = state.to_prompt(self.target, metric_name=self.metric_name, maximize=True)
        validator_src = inspect.getsource(validate_packing)
        return f"""You are an expert in circle packing and computational geometry.
Pack {self.num_circles} circles in the unit square to maximize the sum of radii.

The following read-only validation rule is used:
```python
{validator_src}
```

{state_ctx}

Reason about boundary placement, staggered/hexagonal arrangements, gaps, and
continuous optimization. Define `run_packing()` returning `(centers, radii,
sum_radii)`, where centers has shape ({self.num_circles}, 2) and radii shape
({self.num_circles},). Use numpy/scipy/math, top-level helpers, and no file or
network IO. Return the program inside a ```python code block.
"""

    def evaluate(self, response_text: str, state: State) -> RewardResult:
        code = self.extract_code(response_text)
        if not code:
            return self.failure("no python code block found")
        res = run_python_entrypoint(
            code, self.entrypoint, self.eval_timeout,
            cpus=list(range(self.num_cpus_per_task)),
        )
        if not res.ok:
            return self.failure(res.error, stdout=res.stdout)
        try:
            centers, radii, _reported = res.value
            centers = np.asarray(centers, dtype=float)
            radii = np.asarray(radii, dtype=float)
            if centers.shape != (self.num_circles, 2) or radii.shape != (self.num_circles,):
                raise ValueError("incorrect centers/radii shape")
            if not validate_packing(centers, radii):
                raise ValueError("packing is not valid")
            score = float(np.sum(radii))
        except Exception as exc:  # noqa: BLE001
            return self.failure(f"verification failed: {exc}", stdout=res.stdout)
        return RewardResult(
            reward=score, raw_score=score, correctness=1.0,
            construction=[], msg=f"sum of radii: {score:.8f}", stdout=res.stdout,
            metrics={"sum_radii": score, "num_circles": self.num_circles},
        )


class CirclePacking26Env(_CirclePackingBase):
    num_circles = 26
    target = 2.636


class CirclePacking32Env(_CirclePackingBase):
    num_circles = 32
    target = 2.940
