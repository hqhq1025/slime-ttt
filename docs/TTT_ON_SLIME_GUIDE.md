# How TTT-Discover was ported onto slime (full-parameter)

This guide is the narrative companion to the code: it explains, end to end, how
the **TTT-Discover** test-time-RL algorithm was put on top of **slime v0.3.0** as
**full-parameter** training, what was changed in slime core, what was added as a
plugin, and *why* each decision was made. It assumes no prior knowledge of either
codebase.

- Reference paper/code: *Learning to Discover at Test Time*, arXiv:2601.16175,
  repo `test-time-training/discover` (cloned at `../discover/`).
- Target framework: `THUDM/slime` v0.3.0 (cloned at `../slime/`).

---

## 1. The two systems in one paragraph each

**TTT-Discover.** For a *single* hard problem (e.g. "find a step function
minimizing the Erdős overlap constant"), keep RL-training the LLM at inference
time. Each step: sample several "seed" solutions from an archive, build a prompt
that shows the best-so-far solution, generate a *group* of candidate programs,
run each in a sandbox to get a scalar score, and do a policy-gradient update that
pushes probability toward the **rare best** candidates. The archive then absorbs
the newly discovered solutions (a small tree search), so the next step's prompts
are seeded with better starting points. Two novelties matter: (a) the **entropic
adaptive-β advantage** — instead of GRPO's "reward − mean", it exponentially
up-weights the best in-group samples, because in discovery you only care about
the single best solution; and (b) the **archive + best-so-far prompt
conditioning** that couples in-context search with weight-space RL. The released
code trains **LoRA** through the **Tinker** service.

**slime.** A production RL post-training framework: **Megatron** does
**full-parameter** training, **SGLang** does rollout/generation, and a **Data
Buffer** bridges them. Everything flows through one path:
`generate → convert-to-train-data → train → sync-weights`. It is built to be
extended *without forking the trainer*: you can replace the rollout
(`--rollout-function-path`), the per-sample generation
(`--custom-generate-function-path`), the reward post-processing
(`--custom-reward-post-process-path`), or the advantage computation
(`--custom-advantage-function-path`). slime is full-parameter by default — there
is no training-time LoRA in it (the only `lora` symbols are MLA architecture
params).

The port, in one sentence: **keep TTT-Discover's algorithm, swap Tinker-LoRA for
slime-Megatron-full-parameter, and express the discovery loop through slime's
rollout extension point.**

---

## 2. The mapping (the heart of the port)

| TTT-Discover concept | Where it lives in the reference | How it is realized on slime |
|---|---|---|
| Single-problem dataset (`len==1`) | `SingleProblemDataset` | custom rollout + `--disable-rollout-global-dataset` (no prompt file) |
| Group rollout (`group_size` × `groups_per_batch`) | `do_group_rollout` | `--n-samples-per-prompt` × `--rollout-batch-size` |
| Discovery archive (PUCT tree search) | `PUCTSampler` (Ray detached actor + JSON) | `archive.DiscoveryArchive` inside the slime `RolloutManager` actor |
| Best-so-far prompt conditioning | `State.to_prompt` | `state.State.to_prompt` (faithful port) |
| Sandbox reward | `SandboxRewardEvaluator` + Ray cpu pool | `sandbox.run_python_entrypoint` (subprocess) + env `evaluate` |
| **Entropic adaptive-β advantage** | `compute_advantages` in `rl/train.py` | **slime core**: `--advantage-estimator entropic_adaptive_beta` |
| KL-to-base penalty (`kl_penalty_coef`) | `incorporate_kl_penalty` | `--use-kl-loss --kl-loss-coef` + `--ref-load <base>` |
| Importance-sampling PG loss | Tinker `loss_fn` | slime PPO/GRPO policy loss |
| LoRA training | `create_lora_training_client_async` | **dropped** — slime Megatron full-parameter |
| Two-phase reasoning sampler | `TwoPhaseTokenCompleter` | single-pass SGLang generation (chat template) |
| Drop constant-reward groups | `remove_constant_reward_groups` | unnecessary — entropic zeroes flat groups (optional native filter) |

Two halves fell out of this mapping:

1. **A general slime feature** — the entropic advantage estimator — because it is
   not TTT-specific; it is a new way to turn a group of rewards into advantages.
   This belongs *in slime core*.
2. **A self-contained plugin** — the discovery loop (archive, env, sandbox,
   prompt conditioning) — because it *is* TTT-specific. This belongs in
   `examples/ttt_discover/`.

---

## 3. The core "魔改": a first-class entropic advantage estimator

This is the only change to slime's core, and it is deliberately small:
**+52 lines across 3 files, plus one new module.**

### 3.1 The math (`slime/utils/entropic_advantage.py`, new)

For a group of rewards `r₁…r_k`, form the tilted distribution
`qᵢ ∝ exp(β·rᵢ)` and use a **leave-one-out (LOO)** normalizer so a sample is
never compared to itself:

```
Zᵢ = (Σⱼ exp(β·rⱼ) − exp(β·rᵢ)) / (k − 1)      # LOO baseline
wᵢ = exp(β·rᵢ) / Zᵢ
Aᵢ = wᵢ − 1
```

`β` is solved **per group** by bisection so that `KL(q‖uniform) = target_kl`
(default `log 2`). Intuition:

- A *flat* group (all rewards equal) → `β → 0` → `Aᵢ ≈ 0` → no gradient. This is
  why constant-reward groups are harmless and need not be dropped.
- A group with a clear winner → the winner gets a large positive advantage, the
  losers small negative ones — exactly the "find the best, not the average"
  behaviour discovery needs.
- The target-KL cap stops the weighting collapsing onto one sample.

The module is pure NumPy (no torch/Megatron) so it is CPU-unit-tested
(`examples/ttt_discover/tests/test_entropic_advantage.py`) and is the single
source of truth used by both the controller and the plugin.

### 3.2 Where it plugs into slime's two-stage advantage path

slime computes advantages in **two** places, and we touch both:

1. **Controller side** — `RolloutManager._post_process_rewards`
   (`slime/ray/rollout.py`). This is where GRPO does its `reward − group_mean`.
   We add a branch: if the estimator is `entropic*`, call
   `compute_entropic_advantages(raw_rewards, group_size=n_samples_per_prompt)`
   and return those as the per-sequence advantages. Reshaping by *group size*
   (not batch size) makes it robust to a variable number of groups.

2. **Training side** — `compute_advantages_and_returns`
   (`slime/backends/megatron_utils/loss.py`). The per-sequence scalar must be
   broadcast to every response token. We route `entropic*` through the existing
   `get_grpo_returns(rewards, kl)` (same broadcast grpo uses). No double
   application: the controller produced the advantage; the trainer only spreads
   it across tokens. KL is applied separately via `--use-kl-loss`.

3. **Arguments** — `slime/utils/arguments.py`: add `entropic` and
   `entropic_adaptive_beta` to `--advantage-estimator` choices, plus
   `--adv-entropic-target-kl`, `--adv-entropic-beta`,
   `--adv-entropic-disable-loo`.

Because `entropic` behaves like `grpo` for the trainer, all the other code paths
(critic disabled, log-prob reuse, etc.) already do the right thing — verified by
reading `actor.py` and the post-init validation.

### 3.3 Why a first-class estimator (and a plugin fallback)

Making it first-class is the cleanest "魔改": entropic-vs-mean-baseline is a
general RL knob and reviewers can find it next to grpo/gspo. But to also support
**unmodified** slime, the identical math is exposed as
`reward_post_process.entropic_reward_post_process`, usable via
`--custom-reward-post-process-path` with stock `--advantage-estimator grpo`.

---

## 4. The plugin: the discovery loop (`examples/ttt_discover/`)

The whole test-time loop is a single `--rollout-function-path`:
`ttt_rollout.generate_rollout(args, rollout_id, data_source, evaluation)`.
On each slime step it does:

```
sample parents from archive          # archive.DiscoveryArchive.sample()
  └─ build_prompt(parent)            # env.build_prompt -> state.to_prompt conditioning
  └─ make a group of N Samples       # slime Sample objects, prompt set
generate all via SGLang router       # reuse slime.rollout.sglang_rollout.generate
evaluate each in a sandbox           # env.evaluate -> sandbox.run_python_entrypoint
  └─ sample.reward = scalar score    # what the entropic estimator will center
expand archive with valid children   # archive.update (best-per-parent, dedup, prune)
return list[list[Sample]] groups     # slime computes entropic advantages + trains
```

Component-by-component:

- **`state.State`** — one candidate (code, construction, value, lineage). Its
  `to_prompt` is a faithful port that injects the last program, its before/after
  score, the target, and truncated stdout into the next prompt.
- **`archive.DiscoveryArchive`** — the PUCT tree search, trimmed from the
  reference `PUCTSampler`. `sample(n)` selects parents by
  `Q + c·scale·P·√(1+T)/(1+n)` with lineage de-duplication; `update()` adds the
  best children per parent, dedups by construction, prunes to a buffer cap, and
  persists a JSON snapshot per step (so runs resume).
- **`sandbox.run_python_entrypoint`** — a standalone port of the reference
  subprocess runner: `spawn`, filesystem-write blocking, BLAS thread caps, a hard
  timeout, and process-group SIGKILL cleanup. It returns the entry point's value
  (pickled) and captures stdout for prompt conditioning. (The reference farmed
  these across a Ray CPU pool; here a thread pool inside the rollout actor is
  enough for single node, and the guide notes the `ray.remote` upgrade.)
- **`environment.TTTEnvironment`** — the per-problem contract:
  `create_initial_states`, `build_prompt`, `evaluate`. `envs/erdos.py` is a
  complete worked example (random seeds, the full Erdős prompt, `1/c5_bound`
  reward).
- **`ttt_args.py` + `train_ttt.py`** — the `--ttt-*` flags are registered via
  slime's `parse_args(add_custom_arguments=…)` hook, so the plugin never edits
  core `arguments.py`. `train_ttt.py` is just `train.py` + that hook.

### Why a custom *rollout* rather than a custom *generate*

slime offers both. `--custom-generate-function-path` customizes one sample's
generation (good for multi-turn agents). But TTT needs to own the **whole step**:
which parents to expand, how to group, how to score, and how to update the
archive *between* steps. That is exactly what `--rollout-function-path` controls,
so the discovery loop lives there. Generation itself still reuses slime's proven
token-level `generate` (router routing, logprob capture, truncation handling).

---

## 5. Full-parameter vs. LoRA — what actually differs

This is the crux of the user's request, and the answer is pleasingly small:

- The reference creates a **LoRA** training client:
  `service_client.create_lora_training_client_async(model_name, rank=32)` and
  lets Tinker own the optimizer/sharding.
- slime has **no training-time LoRA**. Its Megatron actor updates **all**
  parameters by construction. So "full-parameter TTT" needs **no special code** —
  it is what you get by routing the TTT rollout + entropic advantage through
  slime's standard training path. We simply *do not* introduce LoRA.

Consequences to be aware of (documented in the run script & plugin README):

- **Learning rate**: LoRA tolerated `4e-5`; full-parameter wants `1e-6`–`5e-7`.
- **Memory/compute**: full-parameter updates a 4B–120B model every step, so the
  example uses TP/PP, recompute, and dynamic batching like slime's own scripts.
- **KL anchor**: with full-parameter updates the policy can drift faster, so the
  KL-to-base term (`--use-kl-loss` + `--ref-load <base>`) matters more; it is the
  slime-idiomatic realization of the paper's `kl_penalty_coef`.

---

## 6. End-to-end data flow

```
                              ┌─────────────────────── examples/ttt_discover ──────────────────────┐
 rollout step k  ──▶  generate_rollout(args, k):                                                     │
                          archive.sample(rollout_batch_size)  ── parents ──▶ env.build_prompt(state) │
                          group = N×Sample(prompt)                                                    │
                          slime.generate(...) ─────────────▶ SGLang router (full-param weights@k)     │
                          env.evaluate(resp) ──▶ sandbox subprocess ──▶ scalar reward + construction  │
                          archive.update(children)            (persist snapshot)                      │
                      returns list[list[Sample]] (reward set) ──────────────────────────────────────┘
                                   │
        ┌──────────────────────────┴── slime core (RolloutManager) ───────────────────────────┐
        │ _post_process_rewards:  entropic LOO adaptive-β  ──▶ per-sequence advantages          │
        │ _convert_samples_to_train_data: tokens, response_lengths, loss_masks, advantages      │
        └───────────────────────────────────────────────┬───────────────────────────────────────┘
                                                         │
        ┌──────────────────────── slime core (Megatron actor) ───────────────────────────┐
        │ compute_advantages_and_returns: broadcast adv → tokens (entropic≈grpo path)      │
        │ policy loss (+ KL-loss to --ref-load) ──▶ FULL-PARAMETER optimizer step          │
        │ update_weights ──▶ push new weights to SGLang  ─────────────────────────────────┘
                                                         │
                                                         ▼  rollout step k+1 (archive now richer)
```

---

## 7. Running and verifying

```bash
# single node, Erdős example, Qwen3-4B, full-parameter:
bash examples/ttt_discover/run-ttt-erdos-qwen3-4B.sh

# CPU sanity (no GPU): the discovery objective's math
python3 examples/ttt_discover/tests/test_entropic_advantage.py
```

What was verified **on CPU** while building this (no GPUs in the dev box):

- the entropic advantage math (constant→0, winner up-weighted, β solves to the
  target KL, contiguous groups independent, LOO behaviour, ragged-input guard);
- the sandbox (normal return, exception surfaced, timeout, fs-write blocked,
  preamble injection);
- the full discovery path on the Erdős env (seed → prompt conditioning → archive
  sample/update/persist → valid & invalid candidate scoring);
- `py_compile` of all edited core files + the plugin.

What inherently needs a **GPU cluster** (slime's own proven paths, wired to
match its contracts but not run here): SGLang generation, Megatron
full-parameter training, and weight sync.

---

## 8. Design decisions & alternatives considered

- **Keep constant-reward groups instead of dropping them.** The entropic
  advantage already gives them ~0 gradient, and keeping them fixes the batch size
  so slime's group reshape stays simple. (Native
  `--dynamic-sampling-filter-path …check_reward_nonzero_std` is available if you
  prefer dropping, since slime 0.3 supports variable global batch size.)
- **Advantage on the controller side, broadcast on the train side.** This mirrors
  GRPO exactly and avoids a bespoke per-token advantage function; the entropic
  scalar is computed once where the whole group is visible.
- **Reward computed in the rollout, not via `rm_hub`.** The archive update needs
  the parsed construction/value anyway, so scoring there keeps everything in one
  place and avoids a second round-trip.
- **Plugin args via `parse_args` hook, not core `arguments.py`.** Keeps
  TTT-specific flags out of slime core; only the *general* entropic estimator is
  promoted to core.
- **Single-pass generation** instead of the paper's two-phase reasoning sampler —
  simpler and model-agnostic; raise `--rollout-max-response-len` (and start from
  the gpt-oss model script) for a faithful reasoning-heavy run.

---

## 9. Exact change set

**slime core (edited, +52/−2 lines):**
- `slime/utils/entropic_advantage.py` *(new)* — the estimator math.
- `slime/utils/arguments.py` — estimator choices + 3 `--adv-entropic-*` args + `import math`.
- `slime/ray/rollout.py` — entropic branch in `_post_process_rewards`.
- `slime/backends/megatron_utils/loss.py` — route `entropic*` through `get_grpo_returns`.

**plugin (new, `examples/ttt_discover/`):** `ttt_rollout.py`, `train_ttt.py`,
`ttt_args.py`, `environment.py`, `envs/erdos.py`, `archive.py`, `state.py`,
`sandbox.py`, `reward_post_process.py`, `run-ttt-erdos-qwen3-4B.sh`, `README.md`,
`tests/test_entropic_advantage.py`.
