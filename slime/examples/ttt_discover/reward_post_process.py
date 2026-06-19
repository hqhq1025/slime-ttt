"""Drop-in entropic reward post-processor for ``--custom-reward-post-process-path``.

Use this when you want the TTT-Discover entropic advantage **without** applying the
core slime edits (i.e. on a stock slime checkout). Wire it up with::

    --advantage-estimator grpo \
    --custom-reward-post-process-path examples.ttt_discover.reward_post_process.entropic_reward_post_process

With ``advantage_estimator=grpo`` the training side broadcasts the per-sequence
advantage we return here to response tokens (``get_grpo_returns``), so the result
is identical to the first-class ``--advantage-estimator entropic_adaptive_beta``
path — it just keeps the math in the plugin instead of in slime core.

If you applied the core edits, prefer the first-class estimator and leave this
unused.
"""

from __future__ import annotations

import math

from slime.utils.entropic_advantage import compute_entropic_advantages


def entropic_reward_post_process(args, samples):
    """Return ``(raw_rewards, advantages)`` using the entropic estimator.

    ``samples`` is the flat, group-contiguous list slime hands to
    ``_post_process_rewards``. We reshape by ``n_samples_per_prompt`` so a
    variable number of groups is fine and constant-reward groups self-zero.
    """
    raw_rewards = [s.get_reward_value(args) for s in samples]
    estimator = getattr(args, "advantage_estimator", "grpo")
    if estimator not in ("entropic", "entropic_adaptive_beta"):
        # When invoked with grpo (the common stock-slime case), default to the
        # adaptive-beta variant from the paper.
        estimator = "entropic_adaptive_beta"
    advantages = compute_entropic_advantages(
        raw_rewards,
        group_size=args.n_samples_per_prompt,
        estimator=estimator,
        beta=getattr(args, "adv_entropic_beta", 2.0),
        target_kl=getattr(args, "adv_entropic_target_kl", math.log(2)),
        leave_one_out=getattr(args, "adv_entropic_leave_one_out", True),
    )
    return raw_rewards, advantages
