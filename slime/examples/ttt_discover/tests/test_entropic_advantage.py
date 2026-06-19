"""CPU unit tests for the entropic (LOO, adaptive-beta) advantage estimator.

Run with ``pytest`` or directly: ``python3 tests/test_entropic_advantage.py``.
These need only numpy — no torch, GPU, Megatron, or SGLang — so they verify the
core "魔改" math in isolation.
"""

import math
import os
import sys

import numpy as np

# Make ``slime`` importable when run as a bare script from the plugin dir.
_SLIME_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _SLIME_ROOT not in sys.path:
    sys.path.insert(0, _SLIME_ROOT)

from slime.utils.entropic_advantage import (  # noqa: E402
    compute_entropic_advantages,
    entropic_group_advantages,
    solve_adaptive_beta,
)


def test_constant_group_is_zero():
    # A group with no reward spread must produce ~0 advantage (no gradient),
    # which is why constant-reward groups are harmless and need not be dropped.
    adv = compute_entropic_advantages([0.5] * 8, group_size=8)
    assert max(abs(a) for a in adv) < 1e-9


def test_winner_is_upweighted_losers_downweighted():
    adv = compute_entropic_advantages([0, 0, 0, 0, 0, 0, 0, 1.0], group_size=8)
    assert adv[-1] > 0
    assert all(a < 0 for a in adv[:-1])


def test_adaptive_beta_hits_target_kl():
    r = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for target in (math.log(2), 0.3, 1.0):
        beta = solve_adaptive_beta(r, target_kl=target)
        # recompute KL(q||uniform) at the solved beta
        logits = beta * (r - r.max())
        logq = logits - (logits.max() + math.log(np.exp(logits - logits.max()).sum()))
        q = np.exp(logq)
        kl = float((q * (logq + math.log(len(r)))).sum())
        assert abs(kl - target) < 1e-3, (beta, kl, target)


def test_groups_are_independent_and_contiguous():
    # Two contiguous groups of size 4: a winner group then a flat group.
    flat = [0, 0, 0, 1.0, 0.5, 0.5, 0.5, 0.5]
    adv = compute_entropic_advantages(flat, group_size=4)
    assert adv[3] > 0 and all(a < 0 for a in adv[:3])  # winner group
    assert all(abs(a) < 1e-9 for a in adv[4:])  # flat group -> 0


def test_leave_one_out_baseline_excludes_self():
    # With LOO, identical rewards give exactly 0 (Z_i == e_i); with the full-group
    # baseline the same holds, but a near-winner is scored against the rest only.
    r = np.array([0.0, 0.0, 1.0])
    loo = entropic_group_advantages(r, beta=2.0, leave_one_out=True)
    full = entropic_group_advantages(r, beta=2.0, leave_one_out=False)
    # LOO up-weights the winner more strongly than the full-group baseline.
    assert loo[2] > full[2]


def test_rejects_ragged_input():
    try:
        compute_entropic_advantages([0.0, 1.0, 2.0], group_size=2)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-multiple length")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} entropic-advantage tests passed.")


if __name__ == "__main__":
    _run_all()
