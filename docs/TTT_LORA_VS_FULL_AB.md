# Qwen3-4B TTT-Discover: LoRA vs full-parameter A/B

Snapshot: 2026-08-18. Task: Erdős minimum overlap. Lower is better.

## Protocol

Both runs used the same model, task prompt, verifier, PUCT implementation,
two-phase generation, and online-search budget:

- Qwen3-4B on 8×A100 80GB;
- 10 TTT updates;
- 8 parent groups × 16 candidates = 128 online rollouts per update;
- 1,280 rollouts total;
- 8,192-token response limit and 4,096-token phase-1 boundary;
- adaptive-entropic advantages and token-level KL shaping.

The adaptation hyperparameters were method appropriate:

- full parameter: Adam, learning rate `1e-6`;
- LoRA: rank 32, alpha 32, dropout 0, QKV/attention-output/MLP projections,
  Adam, learning rate `4e-5`.

The experiment deliberately regenerated trajectories online. Reusing the old
full-parameter trajectories would test an offline update but would not measure
the central TTT effect whereby an updated model changes later exploration.

## Results

| Metric | Full parameter | LoRA r=32 | Difference |
|---|---:|---:|---:|
| Final best objective ↓ | **0.38165901** | 0.38190008 | full better by 0.00024107 |
| Gap to reported 0.380876 ↓ | **0.00078301** | 0.00102408 | full smaller |
| Valid rollouts | 400 / 1,280 | **425 / 1,280** | LoRA +25 |
| Valid rate | 31.25% | **33.20%** | LoRA +1.95 points |
| Final archive size | 137 | **139** | LoRA +2 |
| Valid-score median ↓ | **0.383389** | 0.391874 | full much better |
| Valid-score best 10% cutoff ↓ | **0.382032** | 0.382202 | full better |
| Truncated rollouts | 1 | 1 | tied |
| Mean response length | 4,306.7 | 4,247.0 | similar |

### Best-so-far curve

| Step | Full parameter | LoRA r=32 |
|---:|---:|---:|
| 0 | **0.382321** | 0.384827 |
| 1 | **0.381785** | 0.384827 |
| 2 | **0.381785** | 0.382700 |
| 3 | **0.381785** | 0.382148 |
| 4 | **0.381785** | 0.382148 |
| 5 | **0.381785** | 0.382078 |
| 6 | **0.381760** | 0.382078 |
| 7 | **0.381760** | 0.381900 |
| 8 | **0.381760** | 0.381900 |
| 9 | **0.381659** | 0.381900 |

### Valid rollouts per step

| Step | Full parameter | LoRA r=32 |
|---:|---:|---:|
| 0 | 28 | 25 |
| 1 | 35 | 45 |
| 2 | 50 | 49 |
| 3 | 44 | 47 |
| 4 | 28 | 33 |
| 5 | 38 | 54 |
| 6 | 44 | 44 |
| 7 | 26 | 59 |
| 8 | 52 | 31 |
| 9 | 55 | 38 |

## Interpretation

In this single matched-budget run, rank-32 LoRA learned feasibility more
readily: it produced more valid programs and a slightly larger archive. Full
parameter training produced a substantially better distribution among valid
solutions and a better final extreme value. This supports the task-level
interpretation that low-rank adaptation efficiently shifts broad generation
behavior, while the additional degrees of freedom in full-parameter training
can help concentrate probability on fine numerical improvements.

The difference is small at the best-solution level: LoRA is 0.000241 above the
full-parameter result, about 0.063% of the full result. It is nevertheless
consistent across the best-so-far curve and the valid-score quantiles.

This is one online run per method, not a multi-seed statistical result. Initial
sampling trajectories differ. The historical LoRA run disabled SGLang CUDA
graphs as a workaround for an intermittent startup hang; current launchers no
longer apply that workaround because it made wall-clock comparisons asymmetric.
A stronger causal claim still requires multiple paired seeds and a
learning-rate/rank sweep.

## Evidence

- Full run: `checkpoints/qwen3-4b-ttt-erdos-8x16x10/`
- LoRA run: `checkpoints/qwen3-4b-ttt-erdos-lora-r32-8x16x10/`
- LoRA smoke: `checkpoints/qwen3-4b-ttt-erdos-lora-r32-smoke-v4/`
- Launchers: `local/run_ttt_erdos_lora_smoke.sh` and
  `local/run_ttt_erdos_lora_scale.sh`
