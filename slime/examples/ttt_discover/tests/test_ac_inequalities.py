import math
from types import SimpleNamespace

import numpy as np

from examples.ttt_discover.envs.ac_inequalities import (
    AC1Env,
    AC2Env,
    evaluate_ac1,
    evaluate_ac2,
)


def _args():
    return SimpleNamespace(ttt_eval_timeout=10, ttt_num_cpus_per_task=1, ttt_fail_reward=0.0)


def test_official_constant_sequence_scores():
    # Direct consequences of the official formulas and useful drift sentinels.
    assert math.isclose(evaluate_ac1([1.0] * 10), 2.0, rel_tol=1e-12)
    assert 0.0 < evaluate_ac2([1.0] * 10) < 1.0


def test_long_sequence_uses_equivalent_bounded_verifier():
    # This size crosses the FFT threshold and guards against O(n^2) stalls.
    assert math.isclose(evaluate_ac1([1.0] * 5000), 2.0, rel_tol=1e-12)
    assert 0.0 < evaluate_ac2([1.0] * 5000) < 1.0


def test_ac1_sandbox_and_direction():
    env = AC1Env(_args())
    state = env.create_initial_states(1)[0]
    response = """```python
def propose_candidate(seed=42, budget_s=10, **kwargs):
    return [1.0] * 10
```"""
    result = env.evaluate(response, state)
    assert result.correctness == 1.0
    assert math.isclose(result.raw_score, 2.0)
    assert math.isclose(result.reward, 0.5, rel_tol=1e-7)
    assert env.value_from_score(1.5) == -1.5


def test_ac2_sandbox_and_invalid_rejection():
    env = AC2Env(_args())
    state = env.create_initial_states(1)[0]
    valid = """```python
def construct_function(seed=42, budget_s=10, **kwargs):
    return [1.0] * 10
```"""
    result = env.evaluate(valid, state)
    assert result.correctness == 1.0
    assert result.reward == result.raw_score
    invalid = valid.replace("[1.0] * 10", "[float('nan')]")
    assert env.evaluate(invalid, state).correctness == 0.0


def test_initialization_matches_official_distribution():
    env = AC1Env(_args())
    state = env.create_initial_states(1)[0]
    assert 1000 <= len(state.construction) < 8000
    assert np.unique(state.construction).size == 1
