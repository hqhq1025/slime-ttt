from pathlib import Path
from types import SimpleNamespace

from examples.ttt_discover.envs.ahc058 import AHC058Env, judge_cpp, score_output


ROOT = Path(__file__).resolve().parents[5]
INPUT_DIR = ROOT / "ttt-discover-official/examples/ahc/lib/problems/ahc058/tools/in"
INPUT_0 = (INPUT_DIR / "0000.txt").read_text()


def test_official_state_transition_known_schedule():
    # Upgrade machine 0^0 on turn 0, then idle. For seed 0 A[0]=C[0][0]=1,
    # so the final apple count is 500 and the verifier score is deterministic.
    output = "0 0\n" + "-1\n" * 499
    score, error = score_output(INPUT_0, output)
    assert error == ""
    assert score == round(100_000 * __import__("math").log2(500))


def test_invalid_unaffordable_upgrade_is_zero():
    output = "3 9\n" + "-1\n" * 499
    score, error = score_output(INPUT_0, output)
    assert score == 0
    assert "not enough apples" in error


def test_not_enough_actions_is_rejected():
    score, error = score_output(INPUT_0, "-1\n" * 499)
    assert score == 0
    assert "not enough actions" in error


def test_cpp_judge_smoke_two_public_cases():
    code = r'''
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, L, T; long long K;
    if (!(cin >> N >> L >> T >> K)) return 1;
    vector<long long> A(N), C(N * L);
    for (auto &x : A) cin >> x;
    for (auto &x : C) cin >> x;
    for (int t = 0; t < T; ++t) cout << -1 << '\n';
}
'''
    result = judge_cpp(code, [INPUT_DIR / "0000.txt", INPUT_DIR / "0001.txt"])
    assert result.compile_error == ""
    assert result.accepted == 2
    assert result.mean_score == 0.0  # log2(initial K=1)


def test_environment_uses_official_reward_scaling(monkeypatch):
    monkeypatch.setenv("TTT_AHC058_INPUT_DIR", str(INPUT_DIR))
    monkeypatch.setenv("TTT_AHC058_MAX_CASES", "1")
    env = AHC058Env(SimpleNamespace(
        ttt_eval_timeout=10, ttt_num_cpus_per_task=1, ttt_fail_reward=0.0,
    ))
    state = env.create_initial_states(1)[0]
    code = r'''```cpp
#include <bits/stdc++.h>
using namespace std;
int main() {
  int N,L,T; long long K,x; cin>>N>>L>>T>>K;
  for(int i=0;i<N+N*L;i++) cin>>x;
  cout << "0 0\n";
  for(int t=1;t<T;t++) cout << "-1\n";
}
```'''
    result = env.evaluate(code, state)
    assert result.correctness == 1.0
    assert result.raw_score > 0
    assert result.reward == result.raw_score / 3_000_000.0
