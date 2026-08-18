from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from examples.ttt_discover.envs.trimul_a100 import TriMulA100Env
from examples.ttt_discover.trimul_eval import (
    OFFICIAL_BENCHMARK_CASES,
    OFFICIAL_TEST_CASES,
    cases_for_profile,
    geometric_mean,
)


def _args(**kwargs):
    defaults = dict(
        ttt_eval_timeout=30,
        ttt_num_cpus_per_task=1,
        ttt_fail_reward=0.0,
        ttt_trimul_profile="smoke",
        ttt_trimul_repeats=3,
        ttt_trimul_gpu_device="0",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_official_suite_sizes_and_smoke_coverage():
    assert len(OFFICIAL_TEST_CASES) == 18
    assert len(OFFICIAL_BENCHMARK_CASES) == 7
    tests, benchmarks = cases_for_profile("smoke")
    assert len(tests) == 4
    assert len(benchmarks) == 2
    assert {case["nomask"] for case in tests} == {True, False}
    assert {case["distribution"] for case in tests} == {"normal", "cauchy"}


def test_geometric_mean_matches_ranking_formula():
    assert geometric_mean([4.0, 9.0]) == pytest.approx(6.0)
    with pytest.raises(ValueError):
        geometric_mean([1.0, 0.0])


def test_environment_prompt_warns_about_actual_mask_and_seeds_official_value():
    env = TriMulA100Env(_args())
    state = env.create_initial_states(1)[0]
    prompt = env.build_prompt(state)
    assert state.value == -1_000_000.0
    assert "actual `mask` tensor must always be honored" in prompt
    assert "@triton.jit" in prompt
    assert math.isfinite(state.value)
