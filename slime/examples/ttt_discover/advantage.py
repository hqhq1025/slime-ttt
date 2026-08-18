"""TTT-Discover advantage shaping used by the slime training backend.

The entropic sequence advantage is computed by the rollout manager.  This
module adds the paper's KL-to-base shaping term token by token::

    A_t = A_entropic + lambda * (mean(log pi_rollout - log pi_0)
                                 - (log pi_rollout_t - log pi_0_t))

This follows ``ttt_discover.rl.train.incorporate_kl_penalty`` in the official
implementation.  It is deliberately an advantage term, not slime's separate
KL-loss regularizer.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def shape_advantages(
    rewards: list[float],
    rollout_log_probs: list[torch.Tensor],
    ref_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    kl_coef: float,
    *,
    process_group=None,
) -> list[torch.Tensor]:
    """Return the official token-level, mean-centered KL-shaped advantages."""
    if not (len(rewards) == len(rollout_log_probs) == len(ref_log_probs) == len(loss_masks)):
        raise ValueError("rewards, rollout/ref log-probs, and masks must have equal lengths")
    if not rewards:
        return []

    diffs = []
    total = torch.zeros((), dtype=torch.float64, device=rollout_log_probs[0].device)
    count = torch.zeros((), dtype=torch.float64, device=rollout_log_probs[0].device)
    for sampled, base, mask in zip(rollout_log_probs, ref_log_probs, loss_masks, strict=True):
        if sampled.shape != base.shape or sampled.shape != mask.shape:
            raise ValueError(
                f"shape mismatch: rollout={sampled.shape}, ref={base.shape}, mask={mask.shape}"
            )
        diff = (sampled.float() - base.float()) * mask.float()
        diffs.append(diff)
        total += diff.double().sum()
        count += mask.double().sum()

    if process_group is not None and dist.is_available() and dist.is_initialized():
        packed = torch.stack((total, count))
        dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=process_group)
        total, count = packed[0], packed[1]

    avg_diff = (total / count.clamp_min(1.0)).float()
    output = []
    for reward, diff, mask in zip(rewards, diffs, loss_masks, strict=True):
        base_adv = torch.full_like(diff, float(reward), dtype=torch.float32)
        output.append(base_adv + float(kl_coef) * mask.float() * (avg_diff - diff))
    return output


def compute_ttt_advantages(args, rollout_data) -> None:
    """slime ``--custom-advantage-function-path`` entry point."""
    rollout_log_probs = rollout_data.get("rollout_log_probs")
    ref_log_probs = rollout_data.get("ref_log_probs")
    if not rollout_log_probs:
        raise ValueError("TTT-Discover requires rollout log-probs for importance sampling")
    if not ref_log_probs:
        raise ValueError("TTT-Discover KL shaping requires --ref-load and a nonzero --kl-coef")

    process_group = None
    if dist.is_available() and dist.is_initialized():
        try:
            from megatron.core import parallel_state as mpu

            process_group = mpu.get_data_parallel_group()
        except (ImportError, RuntimeError):
            process_group = None

    advantages = shape_advantages(
        rewards=rollout_data["rewards"],
        rollout_log_probs=rollout_log_probs,
        ref_log_probs=ref_log_probs,
        loss_masks=rollout_data["loss_masks"],
        kl_coef=args.kl_coef,
        process_group=process_group,
    )
    rollout_data["advantages"] = advantages
    rollout_data["returns"] = advantages
