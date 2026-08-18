from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.ttt_discover.envs.mla_decode_a100 import MLADecodeA100Env
from examples.ttt_discover.mla_decode_a100_eval import (
    BENCHMARK_CASES,
    CORRECTNESS_CASES,
    SMOKE_CASES,
    geometric_mean,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


def _args(**kwargs):
    defaults = dict(
        ttt_eval_timeout=30,
        ttt_num_cpus_per_task=1,
        ttt_fail_reward=0.0,
        ttt_mla_repeats=3,
        ttt_mla_gpu_device="0",
        ttt_mla_official_root=os.environ.get(
            "TTT_OFFICIAL_ROOT", str(REPO_ROOT.parent / "ttt-discover-official")
        ),
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_smoke_cases_are_official_small_shapes():
    assert [(case["prefill"], case["seed"]) for case in SMOKE_CASES] == [
        (128, 9247),
        (512, 2197),
    ]
    assert all(case["batchsize"] == 128 and case["dim"] == 7168 for case in SMOKE_CASES)


def test_full_suite_matches_official_task_yml():
    assert [(case["prefill"], case["seed"]) for case in CORRECTNESS_CASES] == [
        (128, 9247),
        (512, 2197),
        (1024, 9107),
        (2048, 5291),
    ]
    assert [(case["prefill"], case["seed"]) for case in BENCHMARK_CASES] == [
        (4096, 9817),
        (6144, 5291),
    ]


def test_geometric_mean():
    assert geometric_mean([4.0, 9.0]) == pytest.approx(6.0)
    with pytest.raises(ValueError):
        geometric_mean([0.0])


def test_prompt_marks_hardware_adaptation_and_contract():
    env = MLADecodeA100Env(_args())
    state = env.create_initial_states(1)[0]
    prompt = env.build_prompt(state)
    assert state.value == -1_000_000.0
    assert "NVIDIA A100 hardware-adaptation smoke" in prompt
    assert "not the official MI300X/H200 leaderboard" in prompt
    assert "qk_nope_head_dim 128" in prompt
    assert "@triton.jit" in prompt


def test_warm_start_uses_measured_a100_seed(monkeypatch):
    monkeypatch.setenv("TTT_MLA_WARM_START", "1")
    env = MLADecodeA100Env(_args())
    state = env.create_initial_states(1)[0]
    assert state.value == pytest.approx(-539.4624114971953)
    assert "def custom_kernel" in state.code
