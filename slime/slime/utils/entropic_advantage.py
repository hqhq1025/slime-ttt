"""Entropic (leave-one-out, adaptive-beta) group advantages.

This is the central numerical novelty of **TTT-Discover** ("Learning to Discover
at Test Time", arXiv:2601.16175) ported into slime as a first-class advantage
estimator. It is what makes *discovery* work: instead of GRPO's mean baseline
(``reward - mean``), which optimizes the *average* response, the entropic
estimator exponentially up-weights the rare, best-in-group samples — exactly the
behaviour you want when you only care about finding the single best solution to
one hard problem.

Given a group of rewards ``r_1..r_k`` we form a tilted distribution
``q_i ∝ exp(beta * r_i)`` and use leave-one-out (LOO) normalization so a sample
is never compared against itself::

    Z_i = (sum_j exp(beta r_j) - exp(beta r_i)) / (k - 1)      # LOO baseline
    w_i = exp(beta r_i) / Z_i
    A_i = w_i - 1

``beta`` is solved per group so that ``KL(q || uniform) == target_kl`` (default
``log 2``). This keeps the weighting from collapsing onto a single sample while
still strongly favouring the best ones, and it adapts automatically to the
group's reward spread (a flat group gets ``beta -> 0`` and therefore ``A ≈ 0``).

The module is intentionally pure-Python/NumPy (no torch, no Megatron) so it can
be unit-tested on CPU and reused both in the rollout controller
(``slime/ray/rollout.py``) and in any ``--custom-reward-post-process-path``
plugin. See ``slime/backends/megatron_utils/loss.py`` for where the resulting
per-sequence advantage is broadcast to response tokens.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "ENTROPIC_ESTIMATORS",
    "solve_adaptive_beta",
    "entropic_group_advantages",
    "compute_entropic_advantages",
]

# Estimator names recognised by slime's ``--advantage-estimator``.
# ``entropic`` uses a fixed ``beta``; ``entropic_adaptive_beta`` solves ``beta``
# per group to hit ``target_kl`` (the TTT-Discover default).
ENTROPIC_ESTIMATORS = ("entropic", "entropic_adaptive_beta")


def _kl_to_uniform(r: np.ndarray, beta: float) -> float:
    """KL(q || uniform) for q_i ∝ exp(beta * r_i) over the k group samples.

    Computed in a numerically stable way (subtract the max before exp). Equal to
    ``log k - H(q)``; it is 0 when ``beta == 0`` and grows as the tilt sharpens.
    """
    k = r.shape[0]
    if k < 2:
        return 0.0
    logits = beta * (r - r.max())
    logq = logits - _logsumexp(logits)
    q = np.exp(logq)
    return float((q * (logq + math.log(k))).sum())


def _logsumexp(x: np.ndarray) -> float:
    m = float(x.max())
    return m + math.log(float(np.exp(x - m).sum()))


def solve_adaptive_beta(
    r: np.ndarray,
    target_kl: float = math.log(2),
    *,
    beta_max: float = 1e6,
    iters: int = 60,
) -> float:
    """Solve for ``beta >= 0`` such that ``KL(q_beta || uniform) ≈ target_kl``.

    Faithful port of TTT-Discover's ``entropic_adaptive_beta`` bracketing: double
    ``hi`` until the KL exceeds the target, then bisect. A degenerate group (all
    rewards equal, or ``k < 2``) returns ``beta = 0`` which yields zero
    advantages downstream.
    """
    r = np.asarray(r, dtype=np.float64)
    k = r.shape[0]
    if k < 2 or float(r.max() - r.min()) < 1e-12:
        return 0.0

    lo, hi = 0.0, 1.0
    if _kl_to_uniform(r, hi) < target_kl:
        # Grow the upper bracket until it overshoots the target KL.
        while hi < beta_max and _kl_to_uniform(r, hi) < target_kl:
            hi *= 2.0
        if _kl_to_uniform(r, hi) < target_kl:
            return float(hi)  # best effort: cannot reach target within beta_max

    # Bisection: keep the smallest beta whose KL is >= target_kl.
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _kl_to_uniform(r, mid) < target_kl:
            lo = mid
        else:
            hi = mid
    return float(hi)


def entropic_group_advantages(
    r: np.ndarray,
    *,
    beta: float,
    leave_one_out: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Leave-one-out entropic advantages for a single group of rewards.

    ``A_i = exp(beta r_i) / Z_i - 1`` where ``Z_i`` is the (LOO) mean of the
    exponentials. With ``leave_one_out=False`` the full-group mean is used.
    """
    r = np.asarray(r, dtype=np.float64)
    k = r.shape[0]
    e = np.exp(beta * (r - r.max()))
    if k == 1:
        z = e
    elif leave_one_out:
        z = (e.sum() - e) / (k - 1)
    else:
        z = np.full_like(e, e.mean())
    w = e / (z + eps)
    return w - 1.0


def compute_entropic_advantages(
    rewards,
    group_size: int,
    *,
    estimator: str = "entropic_adaptive_beta",
    beta: float = 2.0,
    target_kl: float = math.log(2),
    leave_one_out: bool = True,
):
    """Compute entropic advantages for a flat, group-contiguous reward list.

    Args:
        rewards: flat sequence of length ``num_groups * group_size`` with groups
            laid out contiguously (the order slime's rollout controller produces).
        group_size: number of samples per group (``args.n_samples_per_prompt``).
        estimator: ``"entropic_adaptive_beta"`` (solve beta per group) or
            ``"entropic"`` (fixed ``beta``).
        beta: fixed tilt used only when ``estimator == "entropic"``.
        target_kl: KL(q||uniform) target for the adaptive solver.
        leave_one_out: use the LOO baseline (recommended; matches the paper).

    Returns:
        list[float] advantages aligned with ``rewards``.
    """
    flat = np.asarray(list(rewards), dtype=np.float64)
    n = flat.shape[0]
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if n % group_size != 0:
        raise ValueError(
            f"len(rewards)={n} is not a multiple of group_size={group_size}; "
            "groups must be intact and contiguous for the entropic estimator."
        )

    groups = flat.reshape(-1, group_size)
    out = np.empty_like(groups)
    for gi in range(groups.shape[0]):
        g = groups[gi]
        if estimator == "entropic_adaptive_beta":
            b = solve_adaptive_beta(g, target_kl=target_kl)
        elif estimator == "entropic":
            b = beta
        else:
            raise ValueError(f"Unknown entropic estimator: {estimator}")
        out[gi] = entropic_group_advantages(g, beta=b, leave_one_out=leave_one_out)

    return out.reshape(-1).tolist()
