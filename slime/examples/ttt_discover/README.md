# TTT-Discover on slime (full-parameter)

This example ports **"Learning to Discover at Test Time"** (TTT-Discover,
[arXiv:2601.16175](https://arxiv.org/abs/2601.16175)) onto slime, running the
test-time RL *discovery loop* on slime's native **full-parameter** Megatron +
SGLang stack instead of the paper's Tinker **LoRA** reference implementation.

> TTT-Discover keeps RL-training the model *at test time* on a single hard
> problem: it generates many candidate programs, scores them in a sandbox, and
> updates the policy toward the rare best ones — while an archive evolves the
> best-so-far solutions and feeds them back into the prompt. New SOTA on the
> Erdős minimum-overlap constant, GPU kernels, AtCoder heuristics, and scRNA
> denoising.

## What "full-parameter" changes vs. the paper

| | TTT-Discover (reference) | This port |
|---|---|---|
| Trainer | Tinker managed service | slime → **Megatron, all weights** |
| Adaptation | **LoRA** (rank 32) | **Full-parameter** |
| Sampler | Tinker sampling client | SGLang router (slime) |
| Loss | importance-sampling PG | slime PPO/GRPO policy loss |
| Advantage | entropic adaptive-β (LOO) | **same**, added to slime core |
| KL-to-base | folded into advantage | slime `--use-kl-loss` vs. `--ref-load` |

The discovery *algorithm* is preserved; only the training backend changes.

## File map

```
examples/ttt_discover/
├── ttt_rollout.py      # the --rollout-function-path: the whole discovery loop
├── train_ttt.py        # entry point (= train.py + registers --ttt-* args)
├── ttt_args.py         # --ttt-* CLI arguments
├── environment.py      # TTTEnvironment base + RewardResult + code extraction
├── envs/erdos.py       # worked example: Erdős minimum-overlap
├── archive.py          # PUCT tree-search archive over solutions ("discovery")
├── state.py            # a candidate solution + best-so-far prompt conditioning
├── sandbox.py          # subprocess code execution (timeout, fs-write guard)
├── reward_post_process.py  # drop-in entropic advantage for *unmodified* slime
├── run-ttt-erdos-qwen3-4B.sh   # single-node launch script
└── tests/test_entropic_advantage.py   # CPU unit tests (numpy only)
```

The one piece that lives in **slime core** (not here) is the general-purpose
entropic advantage estimator: `slime/utils/entropic_advantage.py`, wired into
`--advantage-estimator entropic_adaptive_beta` (see the integration guide).

## Quick start

```bash
# from the slime repo root, single node:
bash examples/ttt_discover/run-ttt-erdos-qwen3-4B.sh
```

Key flags (full list in `run-ttt-erdos-qwen3-4B.sh`):

```bash
python3 examples/ttt_discover/train_ttt.py \
  --rollout-function-path examples.ttt_discover.ttt_rollout.generate_rollout \
  --ttt-env-path examples.ttt_discover.envs.erdos.ErdosMinOverlapEnv \
  --advantage-estimator entropic_adaptive_beta \  # the discovery objective
  --adv-entropic-target-kl 0.6931 \               # log(2), paper default
  --disable-rollout-global-dataset \              # no prompt file; archive drives prompts
  --rollout-batch-size 8 \                         # groups per step
  --n-samples-per-prompt 16 \                      # group size
  --num-rollout 50 \                               # test-time steps
  --use-kl-loss --kl-loss-coef 0.001 --ref-load <base-ckpt> \
  ... # standard slime Megatron/SGLang args
```

### Argument mapping (paper → slime)

| TTT-Discover | slime flag |
|---|---|
| `group_size` (64) | `--n-samples-per-prompt` |
| `groups_per_batch` (8) | `--rollout-batch-size` |
| `num_epochs` (50) | `--num-rollout` |
| `adv_estimator="entropic_adaptive_beta"` | `--advantage-estimator entropic_adaptive_beta` |
| target KL `log 2` | `--adv-entropic-target-kl 0.6931` |
| `kl_penalty_coef` (0.1) | `--use-kl-loss --kl-loss-coef` + `--ref-load` |
| single-problem dataset | `--disable-rollout-global-dataset` + custom rollout |
| `remove_constant_reward_groups` | not needed (entropic zeroes flat groups); or `--dynamic-sampling-filter-path …check_reward_nonzero_std` |

## Adding your own problem

Subclass `TTTEnvironment` and implement three methods:

```python
from examples.ttt_discover.environment import TTTEnvironment, RewardResult
from examples.ttt_discover.state import State
from examples.ttt_discover.sandbox import run_python_entrypoint

class MyEnv(TTTEnvironment):
    name = "my_problem"; maximize = True; target = 1.0

    def create_initial_states(self, n) -> list[State]: ...      # random seeds
    def build_prompt(self, state: State) -> str: ...            # condition on best-so-far
    def evaluate(self, response_text, state) -> RewardResult:   # parse + sandbox + score
        code = self.extract_code(response_text)
        res = run_python_entrypoint(code, self.entrypoint, self.eval_timeout)
        ...
        return RewardResult(reward=..., raw_score=..., correctness=1.0, construction=...)
```

Then point `--ttt-env-path` at it. See `envs/erdos.py` for a complete example.

## Running on unmodified slime

If you cannot apply the 3 core edits, use the drop-in post-processor with stock
`--advantage-estimator grpo`:

```bash
--advantage-estimator grpo \
--custom-reward-post-process-path examples.ttt_discover.reward_post_process.entropic_reward_post_process
```

This computes the identical entropic advantages from the plugin side.

## Tests

```bash
python3 examples/ttt_discover/tests/test_entropic_advantage.py   # numpy only, no GPU
```

## Notes & limitations

- **Security**: generated code is executed. The sandbox blocks filesystem writes
  and enforces a timeout, but you should still run on an isolated machine/VPN.
- **Two-phase reasoning**: the paper uses a 26k-token reasoning budget for
  gpt-oss; here generation is single-pass under the model's chat template. For a
  faithful gpt-oss run, raise `--rollout-max-response-len` and start from
  `scripts/models/gpt-oss-20B.sh`.
- **Learning rate**: the paper's `4e-5` was for LoRA; full-parameter runs use a
  much smaller LR (`1e-6`–`5e-7`). Tune per model.
- **Sandbox scale**: evaluation runs in a thread pool inside the rollout actor.
  For large multi-node sweeps, wrap `run_python_entrypoint` in `ray.remote`.

## Per-group shared initial state & initial pool (deployment notes)
- Each rollout step: `archive.sample(rollout_batch_size)` PUCT-selects parent states
  ("reuse"); each parent -> one prompt -> `n_samples_per_prompt` samples sharing it
  (same group_index). Entropic advantage reshapes flat rewards by n_samples_per_prompt
  so leave-one-out centering happens WITHIN each shared-state group.
- Initial pool (FrontierCS): `create_initial_states` returns empty seeds (code="",
  value=None) -> step 0 writes C++ from scratch off the bare statement; valid solutions
  enter the archive and seed later prompts. (Erdos env seeds random valid constructions.)
- This mirrors the paper: pool is env-defined via create_initial_state (empty for code
  tasks, random construction for math/optimization).
