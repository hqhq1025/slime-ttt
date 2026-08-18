# TTT-Discover paper-alignment audit

Audit baseline:

- Paper: *Learning to Discover at Test Time*, arXiv:2601.16175.
- Official code: `test-time-training/discover` commit
  `6c40e82dab9d5de7416ac873ad5cd3106084aaed`.
- slime port baseline: `YanagiOrigami/slime-ttt` commit
  `76daafe1d97f8cd1c8152fb89e1d5dd562ba0ed8`.

## Result

The original slime port had the right high-level loop and the correct adaptive
entropic estimator, but it was not a faithful implementation of the paper's
training objective. This workspace revision fixes the objective-level and
search-accounting differences that can be expressed on slime.

| Component | Paper requirement | Audited slime implementation |
|---|---|---|
| Groups | 8 parents x 64 rollouts, shared state within each group | Same layout; local smoke uses a smaller group only for cost |
| Entropic objective | Adaptive beta per group, KL target log(2), LOO normalization | Formula matches official `compute_advantages` |
| KL to base | Mean-centered, token-level term inside the advantage | `advantage.compute_ttt_advantages` matches official formula |
| Policy loss | Importance ratio against sampler log-probs, one full-batch step | `--use-rollout-logprobs`; clipping disabled with infinite bounds |
| Constant groups | Remove before training; retain one if all are constant | Equivalent zero loss masks; first group retained in all-constant case |
| Reuse | PUCT, rank prior, best-child Q, ancestor visits, lineage blocking | Implemented in `DiscoveryArchive` |
| Failed expansion | Still increments visits | Fixed by passing every attempted parent to archive update |
| Archive | Top 2 children per parent, top 1000 globally, retain seeds | Implemented |
| Long reasoning | Reserve final-answer budget and force final transition | Two-phase SGLang continuation with forced tokens masked from loss |
| Truncated code | Final unclosed code block remains parseable | Parser now matches official EOF behavior |

## Deliberate remaining differences

These prevent claiming reproduction of the paper's reported results, even
though the algorithmic objective is aligned:

1. The paper trains rank-32 LoRA on gpt-oss-120b; the local run trains every
   parameter of Qwen3-4B.
2. The paper uses Adam at 4e-5 for LoRA. The full-parameter launch uses 1e-6;
   optimization dynamics therefore differ.
3. Tinker is replaced by Megatron and SGLang. Tokenization, numerical kernels,
   and sampler/learner mismatch can differ.
4. A cheap smoke run uses 1 x 4 rollouts and short generations instead of the
   paper's 8 x 64 rollouts, 50 steps, and long context.
5. The paper text defines one PUCT visit per expanded parent/group. The official
   repository currently updates visits from individual environment rollouts.
   This port follows Appendix A.2 literally: once per selected parent group,
   including failed groups.

Accordingly, a successful smoke test proves the local implementation and
objective execute. A paper-result reproduction still requires gpt-oss-120b,
rank-32 LoRA or a separately justified full-parameter ablation, full batch/context
settings, 50 steps, and comparison against the paper's released result traces.

## Local validation result

An 8 x A100 80GB Qwen3-4B smoke with one parent and four candidates completed
successfully after the fixes:

- 2/4 candidates passed the Erdős verifier;
- all four used two-phase generation and completed without truncation;
- archive size grew from 1 to 3 and the PUCT expansion counter became 1;
- the reward group was non-flat and produced a mean logged advantage of 0.541;
- rollout/train log-prob mismatch was 0 and PPO clip fraction was 0;
- gradient norm was 3.518, followed by optimizer step, weight sync, and a 53GB
  distributed checkpoint save.

This validates a meaningful one-step policy update. It does not establish the
50-step discovery curve or reproduce any paper headline result.
