# CLAUDE.md — working in slime-ttt

Orientation for AI coding agents. Read this before editing.

## What this repo is

Porting **TTT-Discover** (test-time RL, arXiv:2601.16175) onto **slime v0.3.0**
as **full-parameter** training (not the paper's LoRA/Tinker). Full rationale:
`docs/TTT_ON_SLIME_GUIDE.md`. Plugin usage:
`slime/examples/ttt_discover/README.md`.

## Directory orientation

- `discover/` — the cloned **reference** implementation. **Read-only**; study it,
  don't edit it. Key files: `ttt_discover/rl/train.py` (advantage math),
  `tinker_utils/sampler.py` (PUCT archive), `tinker_utils/state.py`
  (`to_prompt`), `environments/sandbox_reward_evaluator.py`.
- `slime/` — the cloned slime framework, **modified** for TTT. This is where work
  happens.
- `docs/`, `README.md`, `CLAUDE.md` — project docs.

## The change set (know this before touching anything)

**slime core — keep edits minimal (only the general entropic estimator):**
- `slime/slime/utils/entropic_advantage.py` *(new)* — the LOO adaptive-β
  advantage math. **Single source of truth**, pure NumPy, CPU-testable.
- `slime/slime/utils/arguments.py` — added `entropic`/`entropic_adaptive_beta`
  choices + `--adv-entropic-*` args + `import math`.
- `slime/slime/ray/rollout.py` — `_post_process_rewards`: entropic branch
  (controller-side advantage).
- `slime/slime/backends/megatron_utils/loss.py` — route `entropic*` through
  `get_grpo_returns` (broadcast scalar advantage to tokens, train-side).

**Plugin — all TTT-specific logic lives here:** `slime/examples/ttt_discover/`
(`ttt_rollout.py` = the `--rollout-function-path`; `archive.py`, `state.py`,
`sandbox.py`, `environment.py`, `envs/erdos.py`, `ttt_args.py`, `train_ttt.py`,
`reward_post_process.py`, `run-ttt-erdos-qwen3-4B.sh`, `tests/`).

## Gotchas (these will bite you)

1. **slime's package is nested.** Repo root is `slime/`; the importable package
   is `slime/slime/`. So `Sample` is at `slime/slime/utils/types.py`. Don't read
   `slime/utils/...` expecting the package.
2. **`load_function` has no colon syntax.** `slime.utils.misc.load_function`
   splits on the **last dot** only — pass slime path args fully dotted
   (`pkg.mod.attr`), never `pkg.mod:attr`. (`--ttt-env-path` normalizes `:`→`.`
   as a courtesy; other slime path args do not.)
3. **slime is full-parameter. Do NOT add training LoRA.** The only `lora` symbols
   in slime are MLA `q_lora_rank`/`kv_lora_rank` (architecture, unrelated).
   "Full-parameter TTT" = run through slime's normal Megatron path.
4. **Entropic advantage is computed once, broadcast once.** Controller side
   (`_post_process_rewards`) makes the per-sequence advantage; train side
   broadcasts it to tokens. Don't add a second centering. KL is separate
   (`--use-kl-loss`).
5. **Constant-reward groups are kept, not dropped** — entropic gives them ~0
   advantage, and keeping them fixes the group reshape (`reshape(-1,
   n_samples_per_prompt)`).
6. **Plugin args go through the `parse_args` hook**, not core `arguments.py`. The
   entry point is `examples/ttt_discover/train_ttt.py` (= `train.py` +
   `add_ttt_arguments`). Launch scripts must call it, not `train.py`.
7. **Keep `examples/ttt_discover/__init__.py` light** (no SGLang import) so env
   classes and CPU tests load without SGLang/torch. `generate_rollout` is reached
   by its full module path, not re-exported from `__init__`.
8. **No GPUs in this dev environment (WSL2).** You cannot run SGLang or Megatron
   here. Validate with CPU tests + `py_compile`; never claim a training run
   succeeded.

## How to verify (CPU only)

```bash
cd slime
# entropic estimator unit tests (numpy only):
python3 examples/ttt_discover/tests/test_entropic_advantage.py
# syntax-check edited core + plugin:
python3 -m py_compile slime/utils/entropic_advantage.py slime/ray/rollout.py \
  slime/backends/megatron_utils/loss.py examples/ttt_discover/*.py \
  examples/ttt_discover/envs/*.py
```
The discovery path (env → prompt → archive → sandbox) is also CPU-runnable; see
the functional-test snippet in `docs/TTT_ON_SLIME_GUIDE.md` §7.

## Conventions

- Match slime's existing code style (it's the host project). Prefer slime's
  extension points over forking its trainer.
- Keep TTT-specific code in the plugin; only *general* RL features belong in
  slime core.
- When changing the advantage math, edit **only** `entropic_advantage.py` (the
  controller, the plugin fallback, and the tests all import it).
- The reference repo (`discover/`) is the ground truth for algorithm fidelity —
  cross-check against it, but don't copy Tinker-specific code.
