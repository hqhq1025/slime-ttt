import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.ttt_discover.ahc_judge import judge_ahc_public, validate_cache
from examples.ttt_discover.envs.ahc039 import AHC039Env


REPO_ROOT = Path(__file__).resolve().parents[4]
CACHE = Path(
    os.environ.get(
        "TTT_AHC_CACHE_DIR", REPO_ROOT.parent / "ttt-storage/ahc-cache/extracted/cache"
    )
)
RECTANGLE = r"""
#include <iostream>
int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    int n, x, y;
    if (!(std::cin >> n)) return 0;
    for (int i = 0; i < 2 * n; ++i) std::cin >> x >> y;
    std::cout << "4\n0 0\n100000 0\n100000 100000\n0 100000\n";
}
"""


@pytest.mark.skipif(not CACHE.exists(), reason="official AHC cache is not installed")
def test_cache_has_all_ahc039_public_inputs():
    tester, inputs = validate_cache(CACHE, "ahc039")
    assert tester.is_file()
    assert len(inputs) == 150


@pytest.mark.skipif(not CACHE.exists(), reason="official AHC cache is not installed")
def test_native_ahc039_public_judge_smoke():
    result = judge_ahc_public(
        RECTANGLE, problem_id="ahc039", cache_dir=CACHE, max_cases=1, time_limit=2.0
    )
    assert result.status == "ok", result.msg
    assert result.passed == result.num_cases == 1
    assert result.raw_score == 1.0


@pytest.mark.skipif(not CACHE.exists(), reason="official AHC cache is not installed")
def test_compile_failure_is_zero_reward():
    result = judge_ahc_public(
        "int main( {", problem_id="ahc039", cache_dir=CACHE, max_cases=1
    )
    assert result.status == "compile_error"
    assert result.raw_score == 0.0


def test_official_seed_is_loaded_without_interpreting_cpp_escapes():
    env = AHC039Env(SimpleNamespace(
        ttt_eval_timeout=10,
        ttt_num_cpus_per_task=1,
        ttt_fail_reward=0.0,
        ttt_ahc_cache_dir=str(CACHE),
        ttt_ahc_reference_root=os.environ.get(
            "TTT_OFFICIAL_ROOT", str(REPO_ROOT.parent / "ttt-discover-official")
        ),
        ttt_ahc_max_cases=1,
        ttt_ahc_time_limit=2.0,
        ttt_ahc_reward_scale=1500.0,
    ))
    state = env.create_initial_states(1)[0]
    assert state.value == 3755.4
    assert 'best_state.poly.size() << "\\n";' in state.code
