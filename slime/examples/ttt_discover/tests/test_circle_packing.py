from types import SimpleNamespace

from examples.ttt_discover.envs.circle_packing import CirclePacking26Env


def test_grid_packing_is_accepted_and_recomputed():
    env = CirclePacking26Env(SimpleNamespace(
        ttt_eval_timeout=10, ttt_num_cpus_per_task=1, ttt_fail_reward=0.0,
    ))
    state = env.create_initial_states(1)[0]
    response = """```python
import numpy as np
def run_packing():
    centers = []
    for i in range(5):
        for j in range(6):
            if len(centers) < 26:
                centers.append(((j + .5) / 6, (i + .5) / 5))
    radii = np.full(26, 0.07)
    return np.array(centers), radii, 999.0
```"""
    result = env.evaluate(response, state)
    assert result.correctness == 1.0
    assert abs(result.raw_score - 1.82) < 1e-12


def test_overlap_is_rejected():
    env = CirclePacking26Env(SimpleNamespace(
        ttt_eval_timeout=10, ttt_num_cpus_per_task=1, ttt_fail_reward=0.0,
    ))
    state = env.create_initial_states(1)[0]
    response = """```python
import numpy as np
def run_packing():
    return np.full((26, 2), .5), np.full(26, .1), 2.6
```"""
    assert env.evaluate(response, state).correctness == 0.0
